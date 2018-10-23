#include "drake/systems/sensors/visualization/sensors_visualization.h"

#include "drake/systems/sensors/image_to_lcm_image_array_t.h"

namespace drake {
namespace systems {
namespace sensors {

systems::lcm::LcmPublisherSystem* ConnectRgbdCameraToDrakeVisualizer(
    DiagramBuilder<double>* builder,
    const systems::OutputPort<double>& color_image_port,
    const OutputPort<double>& depth_image_port,
    const OutputPort<double>& label_image_port,
    drake::lcm::DrakeLcmInterface* lcm) {
  auto image_to_lcm_image_array =
      builder->template AddSystem<ImageToLcmImageArrayT>(
          "color", "depth", "label");
  image_to_lcm_image_array->set_name("converter");

  auto image_array_lcm_publisher = builder->template AddSystem(
      systems::lcm::LcmPublisherSystem::Make<robotlocomotion::image_array_t>(
          "DRAKE_RGBD_CAMERA_IMAGES", lcm));

  image_array_lcm_publisher->set_name("rgbd_publisher");
  image_array_lcm_publisher->set_publish_period(1. / 10 /* 10 fps */);

  builder->Connect(color_image_port,
                  image_to_lcm_image_array->color_image_input_port());

  builder->Connect(depth_image_port,
                  image_to_lcm_image_array->depth_image_input_port());

  builder->Connect(label_image_port,
                  image_to_lcm_image_array->label_image_input_port());

  builder->Connect(image_to_lcm_image_array->image_array_t_msg_output_port(),
                  image_array_lcm_publisher->get_input_port());

  return image_array_lcm_publisher;
}

}  // namespace sensors
}  // namespace systems
}  // namespace drake
