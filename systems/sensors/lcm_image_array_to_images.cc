#include "drake/systems/sensors/lcm_image_array_to_images.h"

#include <vtkImageExport.h>
#include <vtkNew.h>
#include <vtkJPEGReader.h>
#include <zlib.h>

#include "robotlocomotion/image_array_t.hpp"

#include "drake/common/text_logging.h"
#include "drake/common/unused.h"

using robotlocomotion::image_array_t;
using robotlocomotion::image_t;

namespace drake {
namespace systems {
namespace sensors {
namespace {

bool is_color_image(int8_t type) {
  switch (type) {
    case image_t::PIXEL_FORMAT_RGB:
    case image_t::PIXEL_FORMAT_BGR:
    case image_t::PIXEL_FORMAT_RGBA:
    case image_t::PIXEL_FORMAT_BGRA: {
      return true;
    }
    default: {
      break;
    }
  }
  return false;
}

bool image_has_alpha(int8_t type) {
  switch (type) {
    case image_t::PIXEL_FORMAT_RGBA:
    case image_t::PIXEL_FORMAT_BGRA: {
      return true;
    }
    default: {
      break;
    }
  }
  return false;
}

// TODO(sam.creasey) Unfortunately vtkPNGReader always attempt to open a file,
// even when used in the same manner as vtkJPEGReader below.  We should
// eventually find another way to add PNG support.
void DecompressJpeg(const unsigned char* data, int size, void* out) {
  std::vector<char> buffer(size);
  memcpy(buffer.data(), data, size);

  vtkNew<vtkJPEGReader> jpg_reader;
  jpg_reader->SetMemoryBuffer(buffer.data());
  jpg_reader->SetMemoryBufferLength(size);
  jpg_reader->Update();

  vtkNew<vtkImageExport> exporter;
  exporter->SetInputConnection(jpg_reader->GetOutputPort(0));
  exporter->ImageLowerLeftOff();
  exporter->Update();
  exporter->Export(out);
}

}  // namespace

LcmImageArrayToImages::LcmImageArrayToImages()
    : image_array_t_input_port_index_(
          this->DeclareAbstractInputPort(
              "image_array_t", Value<image_array_t>()).get_index()),
      color_image_output_port_index_(
          this->DeclareAbstractOutputPort(
              "color_image", &LcmImageArrayToImages::CalcColorImage)
          .get_index()),
      depth_image_output_port_index_(
          this->DeclareAbstractOutputPort(
              "depth_image", &LcmImageArrayToImages::CalcDepthImage)
          .get_index()) {
  // TODO(sammy-tri) Calculating our output ports can be kinda expensive.  We
  // should cache the images.
}

void LcmImageArrayToImages::CalcColorImage(
    const Context<double>& context, ImageRgba8U* color_image) const {
  const systems::AbstractValue* input =
      this->EvalAbstractInput(context, image_array_t_input_port_index_);
  DRAKE_ASSERT(input != nullptr);
  const auto& images = input->GetValue<image_array_t>();

  // Look through the image array and just grab the first color image.
  const image_t* image = nullptr;
  for (int i = 0; i < images.num_images; i++) {
    if (is_color_image(images.images[i].pixel_format)) {
      image = &images.images[i];
      break;
    }
  }

  if (!image) {
    *color_image = ImageRgba8U();
    return;
  }

  color_image->resize(image->width, image->height);
  const bool has_alpha = image_has_alpha(image->pixel_format);
  ImageRgb8U rgb_image;
  if (!has_alpha) {
    rgb_image.resize(image->width, image->height);
  }

  switch (image->compression_method) {
    case image_t::COMPRESSION_METHOD_NOT_COMPRESSED: {
      if (has_alpha) {
        memcpy(color_image->at(0, 0), image->data.data(), color_image->size());
      } else {
        memcpy(rgb_image.at(0, 0), image->data.data(), rgb_image.size());
      }
      break;
    }
    case image_t::COMPRESSION_METHOD_ZLIB: {
      int status = 0;
      if (has_alpha) {
        unsigned long dest_len = color_image->size();
        status = uncompress(color_image->at(0, 0), &dest_len,
                            image->data.data(), image->size);
      } else {
        unsigned long dest_len = rgb_image.size();
        status = uncompress(rgb_image.at(0, 0), &dest_len,
                            image->data.data(), image->size);
      }
      if (status != Z_OK) {
        drake::log()->error("zlib decompression failed on incoming LCM image");
        *color_image = ImageRgba8U();
        return;
      }
      break;
    }
    case image_t::COMPRESSION_METHOD_JPEG: {
      if (has_alpha) {
        DecompressJpeg(image->data.data(), image->size,
                       color_image->at(0, 0));
      } else {
        DecompressJpeg(image->data.data(), image->size,
                       rgb_image.at(0, 0));
      }
      break;
    }
    default: {
      drake::log()->error("Unsupported LCM compression method: {}",
                          image->compression_method);
      *color_image = ImageRgba8U();
      return;
    }
  }

  if (!has_alpha) {
    for (int x = 0; x < image->width; x++) {
      for (int y = 0; y < image->height; y++) {
        color_image->at(x, y)[0] = rgb_image.at(x, y)[0];
        color_image->at(x, y)[1] = rgb_image.at(x, y)[1];
        color_image->at(x, y)[2] = rgb_image.at(x, y)[2];
        color_image->at(x, y)[3] = 0xff;
      }
    }
  }

  // TODO(sam.creasey) Handle BGR images, or at least error.
}

void LcmImageArrayToImages::CalcDepthImage(
    const Context<double>& context, ImageDepth32F* depth_image) const {
  const systems::AbstractValue* input =
      this->EvalAbstractInput(context, image_array_t_input_port_index_);
  DRAKE_ASSERT(input != nullptr);
  const auto& images = input->GetValue<image_array_t>();

  // Look through the image array and just grab the first depth image.
  const image_t* image = nullptr;
  for (int i = 0; i < images.num_images; i++) {
    if (images.images[i].pixel_format == image_t::PIXEL_FORMAT_DEPTH) {
      image = &images.images[i];
      break;
    }
  }

  if (!image) {
    *depth_image = ImageDepth32F();
    return;
  }

  depth_image->resize(image->width, image->height);

  ImageDepth16U image_16u;
  bool is_32f = false;

  switch (image->channel_type) {
    case image_t::CHANNEL_TYPE_UINT16: {
      is_32f = false;
      image_16u.resize(image->width, image->height);
      break;
    }
    case image_t::CHANNEL_TYPE_FLOAT32: {
      is_32f = true;
      break;
    }
    default: {
      drake::log()->error("Unsupported depth image channel type: {}",
                          image->channel_type);
      *depth_image = ImageDepth32F();
      return;
    }
  }

  switch (image->compression_method) {
    case image_t::COMPRESSION_METHOD_NOT_COMPRESSED: {
      if (is_32f) {
        memcpy(depth_image->at(0, 0), image->data.data(),
               depth_image->size() * depth_image->kPixelSize);
      } else {
        memcpy(image_16u.at(0, 0), image->data.data(),
               image_16u.size() * image_16u.kPixelSize);
      }
      break;
    }
    case image_t::COMPRESSION_METHOD_ZLIB: {
      int status = 0;
      if (is_32f) {
        unsigned long dest_len = depth_image->size() * depth_image->kPixelSize;
        status = uncompress(
            reinterpret_cast<Bytef*>(depth_image->at(0, 0)), &dest_len,
            image->data.data(), image->size);
      } else {
        unsigned long dest_len = image_16u.size() * image_16u.kPixelSize;
        status = uncompress(
            reinterpret_cast<Bytef*>(image_16u.at(0, 0)), &dest_len,
            image->data.data(), image->size);

      }
      if (status != Z_OK) {
        drake::log()->error("zlib decompression failed on incoming LCM image: {}", status);
        *depth_image = ImageDepth32F();
        return;
      }
      break;
    }
    default: {
      drake::log()->error("Unsupported LCM compression method: {}",
                          image->compression_method);
      *depth_image = ImageDepth32F();
      return;
    }
  }

  if (!is_32f) {
    for (int x = 0; x < image->width; x++) {
      for (int y = 0; y < image->height; y++) {
        depth_image->at(x, y)[0] =
            static_cast<float>(image_16u.at(x, y)[0]) / 1e3;
      }
    }
  }
}




}  // namespace sensors
}  // namespace systems
}  // namespace drake
