#ifndef __MASS_MUSCLE_LIMIT_CONSTRAINT_H__
#define __MASS_MUSCLE_LIMIT_CONSTRAINT_H__
#include "Muscle.h"
#include "dart/dart.hpp"

namespace MASS
{
  class Muscle;
}

class MuscleLimitConstraint : public dart::constraint::ConstraintBase
{
public:
  /// Constructor
  explicit MuscleLimitConstraint(MASS::Muscle *muscle);

  /// Destructor
  virtual ~MuscleLimitConstraint();

  //----------------------------------------------------------------------------
  // Property settings
  //----------------------------------------------------------------------------

  /// Set global error reduction parameter
  static void setErrorAllowance(double _allowance);

  /// Get global error reduction parameter
  static double getErrorAllowance();

  /// Set global error reduction parameter
  static void setErrorReductionParameter(double _erp);

  /// Get global error reduction parameter
  static double getErrorReductionParameter();

  /// Set global error reduction parameter
  static void setMaxErrorReductionVelocity(double _erv);

  /// Get global error reduction parameter
  static double getMaxErrorReductionVelocity();

  /// Set global constraint force mixing parameter
  static void setConstraintForceMixing(double _cfm);

  /// Get global constraint force mixing parameter
  static double getConstraintForceMixing();

  MASS::Muscle *GetMuscle() { return mMuscle; }
  double EvalObjective();
  Eigen::VectorXd EvalGradient();
  //----------------------------------------------------------------------------
  // Friendship
  //----------------------------------------------------------------------------

  friend class dart::constraint::ConstraintSolver;
  friend class dart::constraint::ConstrainedGroup;

protected:
  //----------------------------------------------------------------------------
  // Constraint virtual functions
  //----------------------------------------------------------------------------

  // Documentation inherited
  void update() override;

  // Documentation inherited
  void getInformation(dart::constraint::ConstraintInfo *_lcp) override;

  // Documentation inherited
  void applyUnitImpulse(std::size_t _index) override;

  // Documentation inherited
  void getVelocityChange(double *_delVel, bool _withCfm) override;

  // Documentation inherited
  void excite() override;

  // Documentation inherited
  void unexcite() override;

  // Documentation inherited
  void applyImpulse(double *_lambda) override;

  // Documentation inherited
  dart::dynamics::SkeletonPtr getRootSkeleton() const override;

  // Documentation inherited
  bool isActive() const override;

private:
  ///
  MASS::Muscle *mMuscle;
  std::vector<dart::dynamics::Joint *> mJoints;

  ///
  std::vector<dart::dynamics::BodyNode *> mBodyNodes;
  dart::dynamics::SkeletonPtr mSkeleton;

  /// Index of applied impulse
  std::size_t mNumRelatedDofs;

  bool mActive;
  Eigen::VectorXd mdCdTheta;
  double mViolation;
  double mNegativeVel;

  double mUpperBound;
  double mLowerBound;

  double mOldX;
  int mLifeTime;
  /// Global constraint error allowance
  static double mErrorAllowance;

  /// Global constraint error redection parameter in the range of [0, 1]. The
  /// default is 0.01.
  static double mErrorReductionParameter;

  /// Maximum error reduction velocity
  static double mMaxErrorReductionVelocity;

  /// Global constraint force mixing parameter in the range of [1e-9, 1]. The
  /// default is 1e-5
  /// \sa http://www.ode.org/ode-latest-userguide.html#sec_3_8_0
  static double mConstraintForceMixing;
};

#endif