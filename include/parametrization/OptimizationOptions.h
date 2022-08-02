#ifndef PARAMETRIZATION_OPTIMIZATIONOPTIONS_H
#define PARAMETRIZATION_OPTIMIZATIONOPTIONS_H

#include "termination_conditions.h"


namespace parametrization
{

// The struct that contains all options that can be passed to the function
//  map_to.
// Each option explains inline what it does.
//
template<typename Scalar=double>
struct OptimizationOptions {
    //maximum number of iterations in optimization
    int maxIter = 100000;
    
    //what the initial penalty value (areas) is multiplied by
    Scalar mu0 = 1;
    
    //how often to rescale the penalty parameters and gradient bounds,
    // increases whenever a rescale happens, set to zero to never rescale
    int rescaleFrequency = 5;
    //independent of rescale frequency, rescale every iteration for this many
    // initial iterations, set to zero to never use.
    int initialRescale = 5;
    
    //how much to increase the penalty by when rescaling
    Scalar penaltyIncrease = 2.5;
    //how much to decrease the penalty by when rescaling
    Scalar penaltyDecrease = 2.5;
    //size of the discrepancy between primal and dual error that triggers
    // rescale
    Scalar differenceToRescale = 5;
    
    //when the gradient bound b is rescaled it is set to
    // b^2 = bMargin*(1+||grad||^2)
    Scalar bMargin = 5.;
    //turn muMin bounds on and off
    bool minPenaltyBoundsActive = true;
    //turn proximal bounds on and off
    bool proximalBoundsActive = true;
    //at which step should the bounds be turned on? Set to -1 to turn on at
    // initialization
    int turnPenaltyBoundsOnAt = -1;
    int turnProximalBoundsOnAt = -1;
    
    //how to determine when the optimization converged
    TerminationCondition terminationCondition =
    TerminationCondition::PrimalDualErrorNoflips;
    
    //primal absolute error tolerance for termination when terminating with
    // PrimalDualError
    Scalar pAbsTol = 1e-6;
    //primal relative error tolerance for termination when terminating with
    // PrimalDualError
    Scalar pRelTol = 1e-5;
    //dual absolute error tolerance for termination when terminating with
    // PrimalDualError
    Scalar dAbsTol = 1e-6;
    //dual relative error tolerance for termination when terminating with
    // PrimalDualError
    Scalar dRelTol = 1e-5;
    
    //progress tolerance for termination when terminating with ProgressLessThan
    Scalar progressTol = 1e-5;
    
    //target energy for termination with TargetEnergy
    Scalar targetEnergy = 1e-6;
    
    //target number of iterations for termination with NumberOfIterations
    int terminationIter = 100;
};

}

#endif

