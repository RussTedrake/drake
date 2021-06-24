#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "./ibex.h"
#include <Eigen/Core>

#include "drake/common/symbolic.h"

namespace drake {
namespace solvers {
/// Visitor class which converts a Drake's symbolic Formula into a corresponding
/// ibex::ExprCtr.
class IbexConverter {
 public:
  /// Delete the default constructor.
  IbexConverter() = delete;

  /// Constructs a converter from @p variables.
  explicit IbexConverter(
      const Eigen::Ref<const VectorX<symbolic::Variable>>& variables);

  ///@{ Delete copy/move constructors and copy/move assign operations.
  IbexConverter(const IbexConverter&) = delete;
  IbexConverter& operator=(const IbexConverter&) = delete;
  IbexConverter(IbexConverter&&) = delete;
  IbexConverter& operator=(IbexConverter&&) = delete;
  ///@}

  /// Destructor. If `need_to_delete_variables_` is set, delete
  /// `ibex::ExprSymbol*` in `symbolic_var_to_ibex_var_`.
  ~IbexConverter();

  /// Converts @p f into a corresponding IBEX data structure,
  /// ibex::ExprCtr*.
  ///
  /// @note It is the caller's responsibility to delete the return value. So
  /// far, it is always the case that the returned value is used to construct an
  /// `ibex::Function` object. Simply deleting `ibex::ExprCtr` does not destruct
  /// the included `ibex::ExprNode` objects. However, `ibex::Function`'s
  /// destructor does the job. As a result, we should not call `ibex::cleanup`
  /// function explicitly with the return value of this method.
  const ibex::ExprCtr* Convert(const symbolic::Formula& f);

  /// Converts @p e into the corresponding IBEX data structure,
  /// ibex::ExprNode*.
  ///
  /// @note See the above note in `Convert(const Formula& f)`.
  const ibex::ExprNode* Convert(const symbolic::Expression& e);

  const ibex::Array<const ibex::ExprSymbol>& variables() const;

  void set_need_to_delete_variables(bool value);

 private:
  // Visits @p e and converts it into ibex::ExprNode.
  const ibex::ExprNode* Visit(const symbolic::Expression& e);
  const ibex::ExprNode* VisitVariable(const symbolic::Expression& e);
  const ibex::ExprNode* VisitConstant(const symbolic::Expression& e);
  const ibex::ExprNode* VisitRealConstant(const symbolic::Expression& e);
  const ibex::ExprNode* VisitAddition(const symbolic::Expression& e);
  const ibex::ExprNode* VisitMultiplication(const symbolic::Expression& e);
  const ibex::ExprNode* VisitDivision(const symbolic::Expression& e);
  const ibex::ExprNode* VisitLog(const symbolic::Expression& e);
  const ibex::ExprNode* VisitAbs(const symbolic::Expression& e);
  const ibex::ExprNode* VisitExp(const symbolic::Expression& e);
  const ibex::ExprNode* VisitSqrt(const symbolic::Expression& e);
  const ibex::ExprNode* ProcessPow(const symbolic::Expression& base,
                                   const symbolic::Expression& exponent);
  const ibex::ExprNode* VisitPow(const symbolic::Expression& e);
  const ibex::ExprNode* VisitSin(const symbolic::Expression& e);
  const ibex::ExprNode* VisitCos(const symbolic::Expression& e);
  const ibex::ExprNode* VisitTan(const symbolic::Expression& e);
  const ibex::ExprNode* VisitAsin(const symbolic::Expression& e);
  const ibex::ExprNode* VisitAcos(const symbolic::Expression& e);
  const ibex::ExprNode* VisitAtan(const symbolic::Expression& e);
  const ibex::ExprNode* VisitAtan2(const symbolic::Expression& e);
  const ibex::ExprNode* VisitSinh(const symbolic::Expression& e);
  const ibex::ExprNode* VisitCosh(const symbolic::Expression& e);
  const ibex::ExprNode* VisitTanh(const symbolic::Expression& e);
  const ibex::ExprNode* VisitMin(const symbolic::Expression& e);
  const ibex::ExprNode* VisitMax(const symbolic::Expression& e);
  const ibex::ExprNode* VisitIfThenElse(const symbolic::Expression&);
  const ibex::ExprNode* VisitCeil(const symbolic::Expression& e);
  const ibex::ExprNode* VisitFloor(const symbolic::Expression& e);
  const ibex::ExprNode* VisitUninterpretedFunction(const symbolic::Expression&);

  // Visits @p e and converts it into ibex::ibex::ExprNode.
  const ibex::ExprCtr* Visit(const symbolic::Formula& f, bool polarity);
  const ibex::ExprCtr* VisitFalse(const symbolic::Formula&, bool);
  const ibex::ExprCtr* VisitTrue(const symbolic::Formula&, bool);
  const ibex::ExprCtr* VisitVariable(const symbolic::Formula&, bool);
  const ibex::ExprCtr* VisitEqualTo(const symbolic::Formula& f, bool polarity);
  const ibex::ExprCtr* VisitNotEqualTo(const symbolic::Formula& f,
                                       bool polarity);
  const ibex::ExprCtr* VisitGreaterThan(const symbolic::Formula& f,
                                        bool polarity);
  const ibex::ExprCtr* VisitGreaterThanOrEqualTo(const symbolic::Formula& f,
                                                 bool polarity);
  const ibex::ExprCtr* VisitLessThan(const symbolic::Formula& f, bool polarity);
  const ibex::ExprCtr* VisitLessThanOrEqualTo(const symbolic::Formula& f,
                                              bool polarity);
  const ibex::ExprCtr* VisitConjunction(const symbolic::Formula&, bool);
  const ibex::ExprCtr* VisitDisjunction(const symbolic::Formula&, bool);
  const ibex::ExprCtr* VisitNegation(const symbolic::Formula& f, bool polarity);
  const ibex::ExprCtr* VisitForall(const symbolic::Formula&, bool);
  const ibex::ExprCtr* VisitIsnan(const symbolic::Formula&, bool);
  const ibex::ExprCtr* VisitPositiveSemidefinite(const symbolic::Formula&,
                                                 bool);

  // ---------------
  // Member fields
  // ---------------

  // True if we need to delete the variables (ibex::ExprSymbol
  // objects) in the destructor. At initialization, this is true. But
  // when `Convert()` method returns a non-null pointer, it changes to
  // false assuming that the caller will delete the variables.
  bool need_to_delete_variables_{true};

  // Map : symbolic::Variable → ibex::ExprSymbol*.
  std::unordered_map<symbolic::Variable::Id, const ibex::ExprSymbol*>
      symbolic_var_to_ibex_var_;

  ibex::Array<const ibex::ExprSymbol> var_array_;

  // Represents the value `0.0`. We use this to avoid a possible
  // memory leak caused by IBEX code: See
  // https://github.com/ibex-team/ibex-lib/blob/af48e38847414818913b6954e1b1b3050aa14593/src/symbolic/ibex_ExprCtr.h#L53-L55
  const ibex::ExprNode* zero_;

  // Makes VisitFormula a friend of this class so that it can use private
  // operator()s.
  friend const ibex::ExprCtr* ::drake::symbolic::VisitFormula<
      const ibex::ExprCtr*>(IbexConverter*, const symbolic::Formula&,
                            const bool&);
  // Makes VisitExpression a friend of this class so that it can use private
  // operator()s.
  friend const ibex::ExprNode* ::drake::symbolic::VisitExpression<
      const ibex::ExprNode*>(IbexConverter*, const symbolic::Expression&);
};

}  // namespace solvers
}  // namespace drake
