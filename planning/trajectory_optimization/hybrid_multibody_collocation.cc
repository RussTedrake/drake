#include "drake/planning/trajectory_optimization/hybrid_multibody_collocation.h"

#include <limits>

#include "drake/multibody/inverse_kinematics/distance_constraint.h"
#include "drake/multibody/optimization/contact_wrench_evaluator.h"
#include "drake/multibody/optimization/sliding_friction_complementarity_constraint.h"
#include "drake/multibody/optimization/static_friction_cone_constraint.h"
#include "drake/multibody/plant/externally_applied_spatial_force.h"

namespace drake {
namespace planning {
namespace trajectory_optimization {

using Eigen::Map;
using Eigen::MatrixXd;
using Eigen::Vector2d;
using Eigen::VectorBlock;
using Eigen::VectorXd;
using geometry::ProximityProperties;
using geometry::SceneGraphInspector;
using math::AreAutoDiffVecXdEqual;
using multibody::CoulombFriction;
using multibody::Frame;
using multibody::JacobianWrtVariable;
using multibody::MultibodyPlant;
using solvers::MathematicalProgram;
using solvers::MatrixXDecisionVariable;
using solvers::VectorXDecisionVariable;
using systems::Context;
using systems::LeafSystem;
using trajectories::PiecewisePolynomial;

const double kInf = std::numeric_limits<double>::infinity();

using ContactPair = SortedPair<geometry::GeometryId>;
using ContactPairs = std::set<ContactPair>;

namespace internal {

// TODO(russt): Finish implementing caching by applying the chain rule locally
// (as we did in direct collocation).

// Computes the (combined) Coulomb friction for a @p contact_pair.
CoulombFriction<double> ComputeCoulombFriction(
    const SceneGraphInspector<AutoDiffXd>& inspector,
    const ContactPair& contact_pair) {
  // Compute the friction.
  const ProximityProperties& geometryA_props =
      *inspector.GetProximityProperties(contact_pair.first());
  const ProximityProperties& geometryB_props =
      *inspector.GetProximityProperties(contact_pair.second());

  const CoulombFriction<double>& geometryA_friction =
      geometryA_props.GetProperty<CoulombFriction<double>>("material",
                                                           "coulomb_friction");
  const CoulombFriction<double>& geometryB_friction =
      geometryB_props.GetProperty<CoulombFriction<double>>("material",
                                                           "coulomb_friction");

  return CalcContactFrictionFromSurfaceProperties(geometryA_friction,
                                                  geometryB_friction);
}

/* As in StaticFrictionConeConstraint, implement
0 ≤ μ*fᵀn,
fᵀ((1+μ²)nnᵀ - I)f ≥ 0.
We provide this local implementation because using the constraint directly
would invalidate any MbP caching that we've implemented here.
*/
void StaticFrictionConeConstraints(
    const Eigen::Ref<const VectorX<AutoDiffXd>>& f,
    const Eigen::Ref<const Matrix3X<AutoDiffXd>>& nhat,
    const Eigen::Ref<const VectorXd>& mu_squared,
    EigenPtr<VectorX<AutoDiffXd>> constraint) {
  const int num_contacts = nhat.cols();
  DRAKE_DEMAND(constraint->size() == 2 * num_contacts);
  for (int i = 0; i < num_contacts; ++i) {
    const auto fi = f.segment<3>(3 * i);
    const auto nhati = nhat.col(i);
    (*constraint)[2 * i] = fi.dot(nhati);
    (*constraint)[2 * i + 1] =
        fi.dot(((1 + mu_squared[i]) * nhati * nhati.transpose() -
                Eigen::Matrix3d::Identity()) *
               fi);
  }
}

// Calculates the relative position of the signed distance closest points
// between the contact bodies (+ their derivatives), expressed in the world
// frame. The positions (and derivatives) of all contacts are stacked into a
// single tall vector of size (3 * in_contact.size()).
//
// Each output (p, J, and Jdotv) is optional -- they will only be computed if
// they are non-null.
void CalcContact(const MultibodyPlant<AutoDiffXd>& plant,
                 const Context<AutoDiffXd>& context,
                 const ContactPairs& in_contact,
                 EigenPtr<VectorX<AutoDiffXd>> p_CbCa_W = nullptr,
                 MatrixX<AutoDiffXd>* J_v_CbCa_W = nullptr,
                 VectorX<AutoDiffXd>* Jdotv_CbCa_W = nullptr,
                 Matrix3X<AutoDiffXd>* nhat_BA_W = nullptr) {
  int num_contacts = in_contact.size();

  const auto& query_object =
      plant.get_geometry_query_input_port()
          .template Eval<geometry::QueryObject<AutoDiffXd>>(context);
  auto& inspector = query_object.inspector();
  Vector3<AutoDiffXd> p_WCa, p_WCb;
  if (p_CbCa_W) {
    p_CbCa_W->resize(3 * num_contacts);
  }
  Matrix3X<AutoDiffXd> J_v_WCa(3, plant.num_velocities()),
      J_v_WCb(3, plant.num_velocities());
  if (J_v_CbCa_W) {
    J_v_CbCa_W->resize(3 * num_contacts, plant.num_velocities());
  }
  Vector3<AutoDiffXd> Jdotv_WCa, Jdotv_WCb;
  if (Jdotv_CbCa_W) {
    Jdotv_CbCa_W->resize(3 * num_contacts);
  }
  if (nhat_BA_W) {
    nhat_BA_W->resize(3, num_contacts);
  }

  int i = 0;
  for (const auto& c : in_contact) {
    geometry::SignedDistancePair<AutoDiffXd> distance_pair =
        query_object.ComputeSignedDistancePairClosestPoints(c.first(),
                                                            c.second());

    Vector3<AutoDiffXd> p_ACa = inspector.GetPoseInFrame(distance_pair.id_A)
                                    .template cast<AutoDiffXd>() *
                                distance_pair.p_ACa;
    Vector3<AutoDiffXd> p_BCb = inspector.GetPoseInFrame(distance_pair.id_B)
                                    .template cast<AutoDiffXd>() *
                                distance_pair.p_BCb;

    const Frame<AutoDiffXd>& A =
        plant.GetBodyFromFrameId(inspector.GetFrameId(c.first()))->body_frame();
    const Frame<AutoDiffXd>& B =
        plant.GetBodyFromFrameId(inspector.GetFrameId(c.second()))
            ->body_frame();

    if (p_CbCa_W) {
      plant.CalcPointsPositions(context, A, p_ACa, plant.world_frame(), &p_WCa);
      plant.CalcPointsPositions(context, B, p_BCb, plant.world_frame(), &p_WCb);
      p_CbCa_W->segment<3>(3 * i) = p_WCa - p_WCb;
    }

    if (J_v_CbCa_W) {
      plant.CalcJacobianTranslationalVelocity(context, JacobianWrtVariable::kV,
                                              A, p_ACa, plant.world_frame(),
                                              plant.world_frame(), &J_v_WCa);
      plant.CalcJacobianTranslationalVelocity(context, JacobianWrtVariable::kV,
                                              B, p_BCb, plant.world_frame(),
                                              plant.world_frame(), &J_v_WCb);
      J_v_CbCa_W->middleRows<3>(3 * i) = J_v_WCa - J_v_WCb;
    }

    if (Jdotv_CbCa_W) {
      Jdotv_WCa = plant.CalcBiasTranslationalAcceleration(
          context, JacobianWrtVariable::kV, A, p_ACa, plant.world_frame(),
          plant.world_frame());
      Jdotv_WCb = plant.CalcBiasTranslationalAcceleration(
          context, JacobianWrtVariable::kV, B, p_BCb, plant.world_frame(),
          plant.world_frame());
      Jdotv_CbCa_W->segment<3>(3 * i) = Jdotv_WCa - Jdotv_WCb;
    }

    if (nhat_BA_W) {
      nhat_BA_W->col(i) = distance_pair.nhat_BA_W;
    }
    ++i;
  }
}

void SetPlantContext(const MultibodyPlant<AutoDiffXd>& plant,
                     const ContactPairs& in_contact,
                     const Eigen::Ref<const AutoDiffVecXd>& state,
                     const Eigen::Ref<const AutoDiffVecXd>& input,
                     const Eigen::Ref<const AutoDiffVecXd>& force,
                     Context<AutoDiffXd>* context) {
  if (input.size() > 0) {
    const systems::InputPort<AutoDiffXd>& actuation_port =
        plant.get_actuation_input_port();
    if (!actuation_port.HasValue(*context) ||
        !AreAutoDiffVecXdEqual(input, actuation_port.Eval(*context))) {
      actuation_port.FixValue(context, input);
    }
  }
  if (!AreAutoDiffVecXdEqual(state,
                             plant.GetPositionsAndVelocities(*context))) {
    plant.SetPositionsAndVelocities(context, state);
  }
  const systems::InputPort<AutoDiffXd>& generalized_force_port =
      plant.get_applied_generalized_force_input_port();
  MatrixX<AutoDiffXd> J;
  // TODO(russt): Confirm that the FixValue below does not immediately
  // cache-invalidate the computation done in this CalcContact step.
  CalcContact(plant, *context, in_contact, nullptr, &J, nullptr);
  VectorX<AutoDiffXd> tau = J.transpose() * force;
  if (!generalized_force_port.HasValue(*context) ||
      !AreAutoDiffVecXdEqual(tau, generalized_force_port.Eval(*context))) {
    generalized_force_port.FixValue(context, tau);
  }
}

class ConstrainedDirectCollocationConstraint : public solvers::Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ConstrainedDirectCollocationConstraint)

  ConstrainedDirectCollocationConstraint(
      const MultibodyPlant<AutoDiffXd>& plant, const ContactPairs& in_contact,
      Context<AutoDiffXd>* context_sample,
      Context<AutoDiffXd>* context_next_sample,
      Context<AutoDiffXd>* context_collocation,
      bool add_next_sample_constraints = false)
      : Constraint(
            (add_next_sample_constraints ? 6 : 4) * in_contact.size() +
                plant.num_multibody_states() + 
                (add_next_sample_constraints ? 24 : 15) * in_contact.size(),
            1 + (2 * plant.num_multibody_states()) +
                (2 * plant.get_actuation_input_port().size()) +
                (4 * 3 * in_contact.size())),
        plant_(plant),
        in_contact_(in_contact),
        context0_(context_sample),
        context1_(context_next_sample),
        context_collocation_(context_collocation),
        add_next_sample_constraints_(add_next_sample_constraints),
        mu_squared_(in_contact_.size()),
        num_inputs_(plant.get_actuation_input_port().size()),
        num_contacts_(in_contact_.size()) {
    UpdateLowerBound(VectorXd::Zero(num_constraints()));
    VectorXd ub = VectorXd::Zero(num_constraints());
    ub.head((add_next_sample_constraints_ ? 6 : 4) * num_contacts_).array() =
        kInf;
    UpdateUpperBound(ub);

    const auto& query_object =
        plant.get_geometry_query_input_port()
            .template Eval<geometry::QueryObject<AutoDiffXd>>(
                *context_collocation_);
    auto& inspector = query_object.inspector();
    int i = 0;
    for (const ContactPair& c : in_contact_) {
      mu_squared_[i++] = 
          std::pow(ComputeCoulombFriction(inspector, c).static_friction(), 2);
    }
  }

  const ContactPairs& in_contact() const { return in_contact_; }

 private:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override {
    AutoDiffVecXd y_t;
    Eval(x.cast<AutoDiffXd>(), &y_t);
    *y = math::ExtractValue(y_t);
  }

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override {
    DRAKE_DEMAND(x.size() == num_vars());
    y->resize(num_constraints());

    // Extract our input variables:
    // h - current time (breakpoint)
    // x0, x1 state vector at time steps k, k+1
    // u0, u1 input vector at time steps k, k+1
    // f0, f1 is force vector at k, k+1
    // fbar is the force correction (at the collocation)
    // vbar is the velocity correction (at the collocation)
    // xcolbar is the collocation point.
    int index = 0;
    const AutoDiffXd h = x(index++);
    const auto x0 = x.segment(index, plant_.num_multibody_states());
    index += plant_.num_multibody_states();
    const auto x1 = x.segment(index, plant_.num_multibody_states());
    index += plant_.num_multibody_states();
    const auto u0 = x.segment(index, num_inputs_);
    index += num_inputs_;
    const auto u1 = x.segment(index, num_inputs_);
    index += num_inputs_;
    const auto f0 = x.segment(index, 3 * num_contacts_);
    index += 3 * num_contacts_;
    const auto f1 = x.segment(index, 3 * num_contacts_);
    index += 3 * num_contacts_;
    const auto fbar = x.segment(index, 3 * num_contacts_);
    index += 3 * num_contacts_;
    const auto vbar = x.segment(index, 3 * num_contacts_);
    index += 3 * num_contacts_;
    DRAKE_DEMAND(index == num_vars());

    index = 0;
    // We put the friction constraints first just to facilitate setting the
    // bounds in the constructor. (Only the friction constraints have non-zero
    // upper bounds.)
    auto friction0 = y->segment(index, 2 * num_contacts_);
    index += 2 * num_contacts_;
    auto friction1 = y->segment(index, 2 * num_contacts_);
    if (add_next_sample_constraints_) {
      index += 2 * num_contacts_;
    }
    auto friction_col = y->segment(index, 2 * num_contacts_);
    index += 2 * num_contacts_;

    auto collocation = y->segment(index, plant_.num_multibody_states());
    index += plant_.num_multibody_states();
    auto phi0 = y->segment(index, 3 * num_contacts_);
    index += 3 * num_contacts_;
    auto psi0 = y->segment(index, 3 * num_contacts_);
    index += 3 * num_contacts_;
    auto alpha0 = y->segment(index, 3 * num_contacts_);
    index += 3 * num_contacts_;
    auto psicol = y->segment(index, 3 * num_contacts_);
    index += 3 * num_contacts_;
    auto alphacol = y->segment(index, 3 * num_contacts_);
    index += 3 * num_contacts_;

    MatrixX<AutoDiffXd> J;
    VectorX<AutoDiffXd> Jdotv;
    Matrix3X<AutoDiffXd> nhat;

    SetPlantContext(plant_, in_contact_, x0, u0, f0, context0_);
    // TODO(russt): consider using the implicit form of the dynamics.
    AutoDiffVecXd xdot0 = plant_.EvalTimeDerivatives(*context0_).CopyToVector();
    CalcContact(plant_, *context0_, in_contact_, &phi0, &J, &Jdotv, &nhat);
    psi0 = J * x0.tail(plant_.num_velocities());
    alpha0 = J * xdot0.tail(plant_.num_velocities()) + Jdotv;
    StaticFrictionConeConstraints(f0, nhat, mu_squared_, &friction0);

    SetPlantContext(plant_, in_contact_, x1, u1, f1, context1_);
    AutoDiffVecXd xdot1 = plant_.EvalTimeDerivatives(*context1_).CopyToVector();
    if (add_next_sample_constraints_) {
      auto phi1 = y->segment(index, 3 * num_contacts_);
      index += 3 * num_contacts_;
      auto psi1 = y->segment(index, 3 * num_contacts_);
      index += 3 * num_contacts_;
      auto alpha1 = y->segment(index, 3 * num_contacts_);
      index += 3 * num_contacts_;

      CalcContact(plant_, *context1_, in_contact_, &phi1, &J, &Jdotv, &nhat);
      psi1 = J * x1.tail(plant_.num_velocities());
      alpha1 = J * xdot1.tail(plant_.num_velocities()) + Jdotv;
      StaticFrictionConeConstraints(f1, nhat, mu_squared_, &friction1);
    }
    DRAKE_DEMAND(index == num_constraints());

    // Cubic interpolation to get xcol and xdotcol.
    AutoDiffVecXd xcol = 0.5 * (x0 + x1) + h / 8 * (xdot0 - xdot1);
    const AutoDiffVecXd xdotcol = -1.5 * (x0 - x1) / h - .25 * (xdot0 + xdot1);
    const AutoDiffVecXd ucol = 0.5 * (u0 + u1);

    // Collocation constraints
    SetPlantContext(plant_, in_contact_, xcol, ucol, fbar,
                    context_collocation_);
    AutoDiffVecXd g =
        plant_.EvalTimeDerivatives(*context_collocation_).CopyToVector();
    // velocity correction
    CalcContact(plant_, *context_collocation_, in_contact_, nullptr, &J,
                nullptr, nullptr);
    VectorX<AutoDiffXd> qdot_correction(plant_.num_positions());
    plant_.MapVelocityToQDot(*context_collocation_, J.transpose() * vbar,
                             &qdot_correction);
    g.head(plant_.num_positions()) += qdot_correction;
    collocation = xdotcol - g;

    // Note: the DAIRLab code doesn't add these extra constraints on vbar and
    // fbar, but I saw non-physical behavior without them.

    // Constrain that the corrected velocities and accelerations to be on the
    // manifold.
    AutoDiffVecXd v_col(plant_.num_velocities());
    plant_.MapQDotToVelocity(*context_collocation_,
                             g.head(plant_.num_positions()), &v_col);
    psicol = J * v_col;

    xcol.tail(plant_.num_velocities()) = v_col;
    SetPlantContext(plant_, in_contact_, xcol, ucol, fbar,
                    context_collocation_);
    CalcContact(plant_, *context_collocation_, in_contact_, nullptr, &J,
                &Jdotv, &nhat);
    alphacol = J * g.tail(plant_.num_velocities()) + Jdotv;

    friction_col = Vector2d::Zero();
    //StaticFrictionConeConstraints(fbar, nhat, mu_squared_, &friction_col);
  }

  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>&,
              VectorX<symbolic::Expression>*) const override {
    throw std::logic_error(
        "ConstrainedDirectCollocationConstraint does not support symbolic "
        "evaluation.");
  }

  const MultibodyPlant<AutoDiffXd>& plant_;
  const ContactPairs& in_contact_;
  Context<AutoDiffXd>* context0_{};
  Context<AutoDiffXd>* context1_{};
  Context<AutoDiffXd>* context_collocation_{};
  bool add_next_sample_constraints_{};

  VectorXd mu_squared_{};
  int num_inputs_;
  int num_contacts_;
};

class ImpactConstraint : public solvers::Constraint {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(ImpactConstraint)

  ImpactConstraint(const MultibodyPlant<AutoDiffXd>& plant,
                   const ContactPairs& in_contact,
                   double coefficient_of_restitution,
                   Context<AutoDiffXd>* post_impact_context)
      : Constraint(plant.num_velocities() + 5 * in_contact.size(),
                   plant.num_multibody_states() + plant.num_velocities() +
                       3 * in_contact.size()),
        plant_(plant),
        in_contact_(in_contact),
        coefficient_of_restitution_(coefficient_of_restitution),
        context_(post_impact_context),
        mu_squared_(in_contact.size()),
        num_contacts_(in_contact.size()) {
    UpdateLowerBound(VectorXd::Zero(num_constraints()));
    VectorXd ub = VectorXd::Zero(num_constraints());
    ub.tail(2 * num_contacts_).array() = kInf;
    UpdateUpperBound(ub);

    const auto& query_object =
        plant.get_geometry_query_input_port()
            .template Eval<geometry::QueryObject<AutoDiffXd>>(*context_);
    auto& inspector = query_object.inspector();

    int i = 0;
    for (const ContactPair& c : in_contact_) {
      mu_squared_[i++] =
          std::pow(ComputeCoulombFriction(inspector, c).static_friction(), 2);
    }
  }

 private:
  void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
              Eigen::VectorXd* y) const override {
    AutoDiffVecXd y_t;
    Eval(x.cast<AutoDiffXd>(), &y_t);
    *y = math::ExtractValue(y_t);
  }

  void DoEval(const Eigen::Ref<const AutoDiffVecXd>& x,
              AutoDiffVecXd* y) const override {
    DRAKE_DEMAND(x.size() == num_vars());
    y->resize(num_constraints());
    int num_velocities = plant_.num_velocities();

    // Extract our input variables:
    // q+, v+, v-, Lambda
    int index = 0;
    const auto xplus = x.segment(index, plant_.num_multibody_states());
    index += plant_.num_positions();  // only increment by num_positions.
    const auto vplus = x.segment(index, num_velocities);
    index += num_velocities;
    const auto vminus = x.segment(index, num_velocities);
    index += num_velocities;
    const auto Lambda = x.segment(index, 3 * num_contacts_);
    index += 3 * num_contacts_;
    DRAKE_DEMAND(index == num_vars());

    index = 0;
    auto dynamics = y->segment(index, num_velocities);
    index += num_velocities;
    auto velocities = y->segment(index, 3 * num_contacts_);
    index += 3 * num_contacts_;
    auto friction = y->segment(index, 2 * num_contacts_);
    index += 2 * num_contacts_;
    DRAKE_DEMAND(index == num_constraints());

    // Implements the derivation from
    // https://underactuated.csail.mit.edu/multibody.html#impulsive_collision

    VectorX<AutoDiffXd> u(0);
    SetPlantContext(plant_, in_contact_, xplus, u, Lambda, context_);
    MatrixX<AutoDiffXd> J;
    Matrix3X<AutoDiffXd> nhat;
    CalcContact(plant_, *context_, in_contact_, nullptr, &J, nullptr, &nhat);

    MatrixX<AutoDiffXd> M(plant_.num_velocities(), plant_.num_velocities());
    plant_.CalcMassMatrixViaInverseDynamics(*context_, &M);

    MatrixX<AutoDiffXd> tmp = J * M.inverse() * J.transpose();
    VectorX<AutoDiffXd> my_lambda =
        -tmp.completeOrthogonalDecomposition().pseudoInverse() * J * vminus;

    dynamics = M * (vplus - vminus) - J.transpose() * Lambda;
    // Note: When coefficient_of_restitution_ == 0, this constraint is redundant
    // (the mode constraints enforce it).
    velocities = 0 * J * vplus;
/*    
    for (int i = 0; i < num_contacts_; ++i) {
      velocities[3 * i] +=
          coefficient_of_restitution_ * J.row(3 * i).dot(vminus);
    }
*/
    unused(coefficient_of_restitution_);
    StaticFrictionConeConstraints(Lambda, nhat, mu_squared_, &friction);
    /*
    log()->info("impact constraint = {}", fmt_eigen(y->transpose()));
    log()->info(
        "with my lambda    = {}",
        fmt_eigen(
            (M * (vplus - vminus) - J.transpose() * my_lambda).transpose()));
    log()->info("M = \n{}", fmt_eigen(M));
    log()->info("vplus.T = {}", fmt_eigen(vplus.transpose()));
    log()->info("vminus.T = {}", fmt_eigen(vminus.transpose()));
    log()->info("J.T = \n{}", fmt_eigen(J.transpose()));
    log()->info("Lambda.T = {}", fmt_eigen(Lambda.transpose()));
    log()->info("dynamics.T = {}", fmt_eigen(dynamics.transpose()));
    log()->info("nhat.T = {}", fmt_eigen(nhat.transpose()));
    */
  }

  void DoEval(const Eigen::Ref<const VectorX<symbolic::Variable>>&,
              VectorX<symbolic::Expression>*) const override {
    throw std::logic_error(
        "ConstrainedDirectCollocationConstraint does not support symbolic "
        "evaluation.");
  }

  const MultibodyPlant<AutoDiffXd>& plant_;
  const ContactPairs in_contact_{};
  double coefficient_of_restitution_{};
  Context<AutoDiffXd>* context_{};
  VectorXd mu_squared_{};
  int num_contacts_{};
};

}  // namespace internal

HybridMultibodyCollocation::ConstrainedDirectCollocation::
    ConstrainedDirectCollocation(
        const HybridMultibodyCollocation& hybrid, std::string name,
        const ContactPairs& sticking_contact,
        const RobotDiagram<AutoDiffXd>& robot_diagram,
        const systems::Context<AutoDiffXd>& robot_diagram_context,
        int num_time_samples, double minimum_time_step,
        double maximum_time_step, solvers::MathematicalProgram* prog)
    : MultipleShooting(hybrid.num_inputs(), hybrid.num_states(),
                       num_time_samples, minimum_time_step, maximum_time_step,
                       prog),
      name_(name),
      robot_diagram_(robot_diagram),
      robot_diagram_context_(robot_diagram_context.Clone()) {
  const MultibodyPlant<AutoDiffXd>& plant = robot_diagram_.plant();
  const int num_positions = plant.num_positions();
  // Disable all collisions in context_.
  auto& inspector = robot_diagram_.scene_graph().model_inspector();
  Context<AutoDiffXd>& scene_graph_context =
      robot_diagram_.GetMutableSubsystemContext(robot_diagram_.scene_graph(),
                                                robot_diagram_context_.get());
  auto filter_manager = robot_diagram_.scene_graph().collision_filter_manager(
      &scene_graph_context);
  geometry::GeometrySet geometry_set(inspector.GetAllGeometryIds());
  filter_manager.Apply(
      geometry::CollisionFilterDeclaration().ExcludeWithin(geometry_set));

  in_contact_ = sticking_contact;
  int num_contacts = in_contact_.size();

  Context<AutoDiffXd>* plant_context =
      &robot_diagram.GetMutableSubsystemContext(plant,
                                                robot_diagram_context_.get());

  std::vector<Context<AutoDiffXd>*> sample_plant_context(N());
  sample_context_.resize(N());
  for (int i = 0; i < N(); ++i) {
    sample_context_[i] = robot_diagram_context_->Clone();
    sample_plant_context[i] = &robot_diagram.GetMutableSubsystemContext(
        plant, sample_context_[i].get());
  }

  placeholder_force_vars_ =
      this->NewSequentialVariable(3 * num_contacts, "force");

  MatrixXDecisionVariable force_correction = prog->NewContinuousVariables(
      3 * num_contacts, N() - 1, "force_correction");
  MatrixXDecisionVariable velocity_correction = prog->NewContinuousVariables(
      3 * num_contacts, N() - 1, "velocity_correction");

  // Add the dynamic constraints.
  // For N-1 time steps, add a constraint which depends on the breakpoint
  // along with the state and input vectors at that breakpoint and the
  // next.
  for (int i = 0; i < N() - 1; ++i) {
    auto constraint =
        std::make_shared<internal::ConstrainedDirectCollocationConstraint>(
            plant, in_contact_, sample_plant_context[i],
            sample_plant_context[i + 1], plant_context, i == N() - 2);
    this->prog()
        .AddConstraint(constraint,
                       {h_vars().segment<1>(i),
                        x_vars().segment(i * num_states(), num_states() * 2),
                        u_vars().segment(i * num_inputs(), num_inputs() * 2),
                        AllContactForces(i), AllContactForces(i + 1),
                        force_correction.col(i), velocity_correction.col(i)})
        .evaluator()
        ->set_description(
            fmt::format("{} collocation constraint for segment {}", name_, i));
  }

  for (const ContactPair& c : hybrid.GetContactPairCandidates()) {
    if (in_contact_.count(c) == 0) {
      // Add distance >= 0 (non-penetration) for geometries that are not
      // in_contact.
      for (int i = 0; i < N(); ++i) {
        // TODO(russt): having one context per sample might help caching here
        // (if the solver evaluated the constraints out of order). But it need
        // not be the same cache used in the contact constraints, since they
        // will have gradients associated with more variables than just the
        // positions and we have no expectation to be able to share that cache.
        this->prog()
            .AddConstraint(std::make_shared<multibody::DistanceConstraint>(
                               &plant, c, plant_context, 0, kInf),
                           x_vars().segment(i * num_states(), num_positions))
            .evaluator()
            ->set_description(fmt::format("{} non-penetration constraint for "
                                          "geometries {}-{} at index {}",
                                          name_, c.first(), c.second(), i));
      }
    }
  }
}

int HybridMultibodyCollocation::ConstrainedDirectCollocation::ContactIndex(
    const ContactPair& contact) const {
  int index = 0;
  bool found = false;
  for (const auto& p : in_contact_) {
    if (p == contact) {
      found = true;
      break;
    }
    index++;
  }
  if (!found) {
    throw std::runtime_error(fmt::format(
        "Contact between geometries ids {} and {} is not active in this mode.",
        contact.first(), contact.second()));
  }
  return index;
}

solvers::VectorXDecisionVariable
HybridMultibodyCollocation::ConstrainedDirectCollocation::ContactForce(
    const ContactPair& contact) const {
  int contact_index = ContactIndex(contact);
  return placeholder_force_vars_.segment<3>(3 * contact_index);
}

solvers::VectorXDecisionVariable
HybridMultibodyCollocation::ConstrainedDirectCollocation::ContactForce(
    const ContactPair& contact, int index) const {
  int contact_index = ContactIndex(contact);
  return GetSequentialVariableAtIndex("force", index)
      .segment<3>(3 * contact_index);
}

solvers::VectorXDecisionVariable
HybridMultibodyCollocation::ConstrainedDirectCollocation::AllContactForces(
    int index) const {
  return GetSequentialVariableAtIndex("force", index);
}

void HybridMultibodyCollocation::ConstrainedDirectCollocation::DoAddRunningCost(
    const symbolic::Expression& g) {
  // Trapezoidal integration:
  //    sum_{i=0...N-2} h_i/2.0 * (g_i + g_{i+1}), or
  // g_0*h_0/2.0 + [sum_{i=1...N-2} g_i*(h_{i-1} + h_i)/2.0] +
  // g_{N-1}*h_{N-2}/2.0.

  prog().AddCost(SubstitutePlaceholderVariables(g * h_vars()(0) / 2, 0));
  for (int i = 1; i <= N() - 2; i++) {
    prog().AddCost(SubstitutePlaceholderVariables(
        g * (h_vars()(i - 1) + h_vars()(i)) / 2, i));
  }
  prog().AddCost(
      SubstitutePlaceholderVariables(g * h_vars()(N() - 2) / 2, N() - 1));
}

PiecewisePolynomial<double> HybridMultibodyCollocation::
    ConstrainedDirectCollocation::ReconstructInputTrajectory(
        const solvers::MathematicalProgramResult& result) const {
  if (robot_diagram_.plant().num_actuated_dofs() == 0) {
    return PiecewisePolynomial<double>();
  }

  VectorXd times = GetSampleTimes(result);
  std::vector<double> times_vec(N());
  std::vector<MatrixXd> inputs(N());

  for (int i = 0; i < N(); i++) {
    times_vec[i] = times(i);
    inputs[i] = result.GetSolution(input(i));
  }
  return PiecewisePolynomial<double>::FirstOrderHold(times_vec, inputs);
}

PiecewisePolynomial<double> HybridMultibodyCollocation::
    ConstrainedDirectCollocation::ReconstructStateTrajectory(
        const solvers::MathematicalProgramResult& result) const {
  VectorXd times = GetSampleTimes(result);
  std::vector<double> times_vec(N());
  std::vector<MatrixXd> states(N());
  std::vector<MatrixXd> derivatives(N());

  const MultibodyPlant<AutoDiffXd>& plant = robot_diagram_.plant();
  Context<AutoDiffXd>* plant_context =
      &robot_diagram_.GetMutableSubsystemContext(plant,
                                                 robot_diagram_context_.get());

  for (int i = 0; i < N(); i++) {
    times_vec[i] = times(i);
    VectorXd u = result.GetSolution(input(i));
    states[i] = result.GetSolution(state(i));
    VectorXd force = result.GetSolution(AllContactForces(i));
    internal::SetPlantContext(plant, in_contact_, states[i].cast<AutoDiffXd>(),
                              u.cast<AutoDiffXd>(), force.cast<AutoDiffXd>(),
                              plant_context);
    derivatives[i] = math::ExtractValue(
        plant.EvalTimeDerivatives(*plant_context).CopyToVector());
  }
  return PiecewisePolynomial<double>::CubicHermite(times_vec, states,
                                                   derivatives);
}

PiecewisePolynomial<double> HybridMultibodyCollocation::
    ConstrainedDirectCollocation::ReconstructContactForceTrajectory(
        const solvers::MathematicalProgramResult& result,
        const ContactPair& contact) const {
  VectorXd times = GetSampleTimes(result);
  if (in_contact_.count(contact) == 0) {
    return PiecewisePolynomial<double>::ZeroOrderHold(
        Vector2d(times(0), times.tail<1>()[0]), MatrixXd::Zero(3, 2));
  }

  std::vector<double> times_vec(N());
  std::vector<MatrixXd> force(N());

  for (int i = 0; i < N(); i++) {
    times_vec[i] = times(i);
    force[i] = result.GetSolution(ContactForce(contact, i));
  }
  return PiecewisePolynomial<double>::FirstOrderHold(times_vec, force);
}

HybridMultibodyCollocation::HybridMultibodyCollocation(
    const RobotDiagram<double>& robot_diagram,
    const Context<double>& robot_diagram_context, double minimum_time_step,
    double maximum_time_step)
    : robot_diagram_(robot_diagram),
      robot_diagram_ad_(systems::System<double>::ToAutoDiffXd(robot_diagram)),
      robot_diagram_context_ad_(robot_diagram_ad_->CreateDefaultContext()),
      minimum_time_step_{minimum_time_step},
      maximum_time_step_(maximum_time_step) {
  if (!robot_diagram_context.has_only_continuous_state()) {
    throw std::logic_error(
        "HybridMultibodyCollocation: The MultibodyPlant in "
        "`robot_diagram` must be in continuous-time mode (time_step=0).");
  }
  robot_diagram_context_ad_->SetTimeStateAndParametersFrom(
      robot_diagram_context);
}

HybridMultibodyCollocation::ContactPairs
HybridMultibodyCollocation::GetContactPairCandidates() const {
  auto collision_candidates = model_inspector().GetCollisionCandidates();
  ContactPairs contact_pairs;
  contact_pairs.insert(collision_candidates.begin(),
                       collision_candidates.end());
  return contact_pairs;
}

HybridMultibodyCollocation::ConstrainedDirectCollocation*
HybridMultibodyCollocation::AddMode(std::string name, int num_time_samples,
                                    const ContactPairs& sticking_contact) {
  ContactPairs in_contact(sticking_contact);

  ContactPairs candidates = GetContactPairCandidates();
  for (const ContactPair& p : in_contact) {
    if (candidates.count(p) == 0) {
      throw std::runtime_error(
          fmt::format("Contact between geometries {} and {} is not in "
                      "the GetContactPairCandidates.",
                      p.first(), p.second()));
    }
  }

  const ConstrainedDirectCollocation* prev{nullptr};
  if (!dircon_.empty()) {
    prev = dircon_.back().get();
    for (const ContactPair& p : in_contact) {
      if (dircon_.back()->in_contact().count(p) == 0) {
        throw std::runtime_error(fmt::format(
            "Contact between geometries {} and {} was not active in the most "
            "recently added mode. The AddMode method can only be used if the "
            "ContactPairs are a strict subset of the ContactPairs from the "
            "previous mode. Use e.g. AddModeWithInelasticImpact to create new "
            "contacts.",
            p.first(), p.second()));
      }
    }
  }

  auto& d = dircon_.emplace_back(std::unique_ptr<ConstrainedDirectCollocation>(
      new ConstrainedDirectCollocation(
          *this, name, sticking_contact, *robot_diagram_ad_,
          *robot_diagram_context_ad_, num_time_samples, minimum_time_step_,
          maximum_time_step_, &prog_)));

  const MultibodyPlant<AutoDiffXd>& plant = robot_diagram_ad_->plant();
  if (prev) {
    // Add continuity constraints with the previous mode.
    prog_.AddLinearEqualityConstraint(prev->final_state() ==
                                      d->initial_state());
    if (plant.get_actuation_input_port().size() > 0) {
      prog_.AddLinearEqualityConstraint(prev->input(prev->num_samples() - 1) ==
                                        d->input(0));
    }
    // TODO(russt): Consider adding force continuity constraints.
  }

  return d.get();
}

HybridMultibodyCollocation::ConstrainedDirectCollocation*
HybridMultibodyCollocation::AddModeWithInelasticImpact(
    std::string name, int num_time_samples,
    const ContactPairs& sticking_contact) {
  // TODO(russt): Add support for sliding friction.

  if (dircon_.empty()) {
    throw std::runtime_error(
        "There are no modes defined yet. You must call AddMode at least once "
        "before you can create a mode with an impact transition using "
        "AddModelWithInelasticImpact.");
  }
  const ConstrainedDirectCollocation& prev = *dircon_.back();

  ContactPairs candidates = GetContactPairCandidates();
  for (const ContactPair& p : sticking_contact) {
    if (candidates.count(p) == 0) {
      throw std::runtime_error(
          fmt::format("Contact between geometries {} and {} is not in "
                      "the GetContactPairCandidates.",
                      p.first(), p.second()));
    }
  }

  auto& d = dircon_.emplace_back(std::unique_ptr<ConstrainedDirectCollocation>(
      new ConstrainedDirectCollocation(
          *this, name, sticking_contact, *robot_diagram_ad_,
          *robot_diagram_context_ad_, num_time_samples, minimum_time_step_,
          maximum_time_step_, &prog_)));

  const MultibodyPlant<AutoDiffXd>& plant = robot_diagram_ad_->plant();

  // Add continuity/impact constraints with the previous mode.
  prog_.AddLinearEqualityConstraint(
      prev.final_state().head(plant.num_positions()) ==
      d->initial_state().head(plant.num_positions()));
  // Define a variable for the impulse, expressed in the world frame.
  auto Lambda = prog_.NewContinuousVariables(
      3 * sticking_contact.size(),
      fmt::format("Lambda impact from mode {} to {}", dircon_.size() - 2,
                  dircon_.size() - 1));
  const double coefficient_of_restitution{0.0};
  Context<AutoDiffXd>* plant_context =
      &robot_diagram_ad_->GetMutableSubsystemContext(
          plant, d->mutable_sample_context(0));
  prog_
      .AddConstraint(std::make_shared<internal::ImpactConstraint>(
                         plant, sticking_contact, coefficient_of_restitution,
                         plant_context),
                     {d->initial_state(),
                      prev.final_state().tail(plant.num_velocities()), Lambda})
      .evaluator()
      ->set_description(fmt::format("impact constraint from mode {} to {}",
                                    dircon_.size() - 2, dircon_.size() - 1));

  if (plant.get_actuation_input_port().size() > 0) {
    prog_.AddLinearEqualityConstraint(prev.input(prev.num_samples() - 1) ==
                                      d->input(0));
  }
  return d.get();
}

PiecewisePolynomial<double>
HybridMultibodyCollocation::ReconstructInputTrajectory(
    const solvers::MathematicalProgramResult& result) const {
  if (dircon_.empty() || num_inputs() == 0) {
    return PiecewisePolynomial<double>();
  }

  auto iter = dircon_.begin();
  PiecewisePolynomial<double> pp = (*iter)->ReconstructInputTrajectory(result);
  ++iter;

  for (; iter != dircon_.end(); ++iter) {
    PiecewisePolynomial<double> this_pp =
        (*iter)->ReconstructInputTrajectory(result);
    // TODO(russt): It could be better to add constraints for the times in the
    // optimization.
    this_pp.shiftRight(pp.end_time() - this_pp.start_time());
    pp.ConcatenateInTime(this_pp);
  }

  return pp;
}

PiecewisePolynomial<double>
HybridMultibodyCollocation::ReconstructStateTrajectory(
    const solvers::MathematicalProgramResult& result) const {
  if (dircon_.empty()) {
    return PiecewisePolynomial<double>();
  }

  auto iter = dircon_.begin();
  PiecewisePolynomial<double> pp = (*iter)->ReconstructStateTrajectory(result);
  ++iter;

  for (; iter != dircon_.end(); ++iter) {
    PiecewisePolynomial<double> this_pp =
        (*iter)->ReconstructStateTrajectory(result);
    this_pp.shiftRight(pp.end_time() - this_pp.start_time());
    pp.ConcatenateInTime(this_pp);
  }

  return pp;
}

PiecewisePolynomial<double>
HybridMultibodyCollocation::ReconstructContactForceTrajectory(
    const solvers::MathematicalProgramResult& result,
    const ContactPair& contact) const {
  if (dircon_.empty()) {
    return PiecewisePolynomial<double>();
  }

  auto iter = dircon_.begin();
  PiecewisePolynomial<double> pp =
      (*iter)->ReconstructContactForceTrajectory(result, contact);
  ++iter;

  for (; iter != dircon_.end(); ++iter) {
    PiecewisePolynomial<double> this_pp =
        (*iter)->ReconstructContactForceTrajectory(result, contact);
    this_pp.shiftRight(pp.end_time() - this_pp.start_time());
    pp.ConcatenateInTime(this_pp);
  }

  return pp;
}

}  // namespace trajectory_optimization
}  // namespace planning
}  // namespace drake
