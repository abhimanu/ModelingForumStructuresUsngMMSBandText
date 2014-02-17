#ifndef IBM_MMSB_HPP_
#define IBM_MMSB_HPP_

#include <cmath>
#include <sstream>
#include <vector>

#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>

#include "Base.hpp"
#include "Random.hpp"

namespace ibm {

// Sampler for MMSB and modNegLinkLL-MMSB
class MmsbSampler {

protected:

// Typedefs and enums
  
  typedef boost::tuple<uint32,uint32> Phi_t;
  enum probMode_t {PRIOR, POSTERIOR}; // Specifies a probability type to compute
  enum sMode_t {SR_MODE, SL_MODE};    // Specifies either sR_ (donor) or sL_
                                      // (receiver) community indicators
  
// Member variables
  
  // Pointers to observed and latent variables
  bool* E_;     // Adjacency matrix
  uint16* sR_;  // Donor levels
  uint16* sL_;  // Receiver levels; sR_(i,j) and sL_(i,j) generate E(i,j).
                // Note that sR_(i,j) is associated with sCounts(i,*),
                // while sL_(i,j) is associated with sCounts(j,*).
  
  // Model parameters
  double alpha_;            // MM vector prior
  double lambda0_;          // -ve link prior
  double lambda1_;          // +ve link prior
  const size_t N_;          // Number of entities
  const size_t K_;          // Number of communities
  const bool modNegLinkLL_; // If false, we use regular MMSB. If true,
                            // we assume a modified MMSB, in which the
                            // likelihood for negative links is always 1.
  
  // Sufficient statistics
  uint32Vec sCounts_;       // N_xK_, sufficient statistics for theta (MM
                            // vectors)
  std::vector<Phi_t> Phi_;  // K_xK_, community-compatibility matrix, Phi_(a,b)
                            // is the link probability from community a to b
  
  // Temporary storage
  doubleVec sProbs_;
  
  // RNG object
  Random rng_;
  
// Functions
  
  // Returns a reference to the element of E_ corresponding to E(i,j)
  bool& getEElement(size_t i, size_t j) {
    return const_cast<bool&>(getEElementHelper(i,j));
  }
  const bool& getEElement(size_t i, size_t j) const {
    return getEElementHelper(i,j);
  }
  const bool& getEElementHelper(size_t i, size_t j) const {
    return E_[i*N_+j];
  }
  
  // Returns references to elements of sR_ or sL_ corresponding to E(i,j),
  // i.e. sR_(i,j) or sL_(i,j)
  uint16& getSElement(size_t i, size_t j, sMode_t sMode) {
    return const_cast<uint16&>(getSElementHelper(i,j,sMode));
  }
  const uint16& getSElement(size_t i, size_t j, sMode_t sMode) const {
    return getSElementHelper(i,j,sMode);
  }
  const uint16& getSElementHelper(size_t i, size_t j, sMode_t sMode) const {
    return (sMode == SR_MODE) ? (sR_[i*N_+j]) : (sL_[i*N_+j]);
  }
  
  // Returns a reference to (i,a)-th element of sCounts_, i.e. the
  // number of sR_ or sL_ belonging to entity i and equal to a.
  uint32& getSCountsElement(size_t i, size_t a) {
    return const_cast<uint32&>(getSCountsElementHelper(i,a));
  }
  const uint32& getSCountsElement(size_t i, size_t a) const {
    return getSCountsElementHelper(i,a);
  }
  const uint32& getSCountsElementHelper(size_t i, size_t a) const {
    return sCounts_[i*K_+a];
  }
  
  // Returns a reference to the element of Phi_ corresponding to E(i,j),
  // based on the current values of sR_ and sL.
  Phi_t& getPhiElement(size_t i, size_t j) {
    return const_cast<Phi_t&>(getPhiElementHelper(i,j));
  }
  const Phi_t& getPhiElement(size_t i, size_t j) const {
    return getPhiElementHelper(i,j);
  }
  const Phi_t& getPhiElementHelper(size_t i, size_t j) const {
    return Phi_[getSElement(i,j,SR_MODE)*K_ + getSElement(i,j,SL_MODE)];
  }
  
  // Computes log(P(E(*,*)|s)). Used to compute the marginal log likelihood of
  // the data E_, or the complete log likelihood log(P(E,s)). This function is
  // NOT used for sampling.
  //
  // Assumes that Phi_ accounts for ALL edges E.
  //
  // All required information is stored in Phi_, so this has runtime polynomial
  // in K_ rather than N_.
  //
  // If modNegLinkLL_ is true, we compute log(P(E+(*)|s)), i.e. the log
  // likelihood of the +ve links, under the "moving -ve link prior" model.
  // Because the likelihood of the -ve links is 1, this is equivalent to
  // log(P(E,s)) under said model.
  double computeELogLikelihood() const {
    double ll = 0.0;
    if (!modNegLinkLL_) {
      // Standard MMSB link log likelihood
      for (size_t a = 0; a < K_; ++a) {
        for (size_t b = 0; b < K_; ++b) {
          const Phi_t& PhiEl = Phi_[a*K_ + b];
          ll +=
            // Inverse prior normalizer
              Random::lnGamma(lambda0_ + lambda1_)
            - Random::lnGamma(lambda0_)
            - Random::lnGamma(lambda1_)
            // Likelihood normalizer
            + Random::lnGamma(PhiEl.get<0>() + lambda0_)
            + Random::lnGamma(PhiEl.get<1>() + lambda1_)
            - Random::lnGamma(PhiEl.get<0>() + PhiEl.get<1>()
                              + lambda0_ + lambda1_);
        }
      }
    } else {
      // Moving -ve link prior model: assume -ve links have likelihood 1, and
      // compute the log of
      // P(E(*,*)|s) = P(E-(*),E+(*)|s) = P(E-(*)) * P(E+(*)|s) = P(E+(*)|s)
      // i.e. the log likelihood of the +ve links. Effectively, the -ve links
      // become part of the prior, due to the way P(s) is defined.
      for (size_t a = 0; a < K_; ++a) {
        for (size_t b = 0; b < K_; ++b) {
          const Phi_t& PhiEl = Phi_[a*K_ + b];
          ll +=
            // Inverse prior normalizer (which includes -ve links)
              Random::lnGamma(PhiEl.get<0>() + lambda0_ + lambda1_)
            - Random::lnGamma(PhiEl.get<0>() + lambda0_)
            - Random::lnGamma(lambda1_)
            // Likelihood normalizer
            + Random::lnGamma(PhiEl.get<0>() + lambda0_)
            + Random::lnGamma(PhiEl.get<1>() + lambda1_)
            - Random::lnGamma(PhiEl.get<0>() + PhiEl.get<1>()
                              + lambda0_ + lambda1_);
        }
      }
    }
    return ll;
  }
  
  // Computes log(P(E(i,j)|s,E_{-(i,j)})). Used for sampling sR_ or sL_.
  //
  // Assumes that the sufficient statistics for E(i,j) are NOT in Phi_.
  //
  // When modNegLinkLL_ is true and E(i,j)==0, we return 0 (likelihood 1).
  double computeELogLikelihood(size_t i, size_t j) const {
    const bool& EEl = getEElement(i,j);
    if (modNegLinkLL_ && EEl == false) {
      return 0;
    } else {
      const Phi_t& PhiEl = getPhiElement(i,j);
      return
          log( (EEl==false)*(PhiEl.get<0>()+lambda0_)
              + (EEl==true)*(PhiEl.get<1>()+lambda1_) )
        - log( PhiEl.get<0>() + PhiEl.get<1>() + lambda0_ + lambda1_);
    }
  }
  
  // Computes log(P(s(*))), or log(P(s(i)) (i.e. the log prior probability for
  // entity i's community indicators sR_(i,*), sL_(*,i)). Used to compute the
  // complete log likelihood log(P(E,s)). These functions are NOT used for
  // sampling.
  //
  // Assumes that sCounts_ accounts for ALL sR_ and sL_.
  double computeSLogPrior() const {
    double ll = 0.0;
    for (size_t i = 0; i < N_; ++i) {
      ll += computeSLogPrior(i);
    }
    return ll;
  }
  double computeSLogPrior(size_t i) const {
    double ll = 0.0;
    double top = 0.0;
    double bot = 0.0;
    for (size_t a = 0; a < K_; ++a) {
      top += alpha_;
      bot += alpha_ + getSCountsElement(i,a);
      ll  += Random::lnGamma(alpha_ + getSCountsElement(i,a))
             - Random::lnGamma(alpha_);
    }
    ll += Random::lnGamma(top) - Random::lnGamma(bot);
    return ll;
  }
  
  // Helper function to sample the element of sR_ or sL_ associated with
  // E(i,j). Used by gsS() and generativeS().
  void sampleSHelper(size_t i, size_t j, sMode_t sMode, probMode_t probMode) {
    uint16& SEl = getSElement(i,j,sMode);
    size_t entity = (sMode == SR_MODE) ? i : j;
    
    // Compute unnormalized prior/posterior probabilities for communities 1 to K
    for (size_t a = 0; a < K_; ++a) {
      // Prior log probability
      sProbs_[a] = log(alpha_ + getSCountsElement(entity,a));
      // Posterior log probability
      if (probMode == POSTERIOR) {
        // Because computeELogLikelihood(i,j) reads the level directly from sR_
        // and sL_, we set SEl so it computes the value conditioned on SEl = a.
        SEl = static_cast<uint16>(a);
        sProbs_[a] += computeELogLikelihood(i,j);
      }
    }
    
    // Sample a new value for SEl
    Random::logprobsToRelprobs(sProbs_,0,K_);
    SEl = static_cast<uint16>(rng_.randDiscrete(sProbs_,0,K_));
  }
  
public:

  // Constructor
  MmsbSampler(bool *Ei, uint16 *sRi, uint16 *sLi,
              double alphai, double lambda0i, double lambda1i,
              size_t Ni, size_t Ki, bool modNegLinkLLi,
              uint32 rngSeedi)
    : // Observed and latent variables
      E_(Ei), sR_(sRi), sL_(sLi),
      // Model parameters
      alpha_(alphai), lambda0_(lambda0i), lambda1_(lambda1i),
      N_(Ni), K_(Ki), modNegLinkLL_(modNegLinkLLi),
      // Sufficient statistics
      sCounts_(N_*K_,0), Phi_(K_*K_,Phi_t(0,0)),
      // Temporary storage
      sProbs_(K_),
      // RNG object
      rng_(rngSeedi)
  { }
  
  // Functions to initialize sufficient statistics based on s*_ and E_
  void initializeSS() {
    initializeSSLatent();
    initializeSSObserved();
  }
  void initializeSSLatent() {
    // Add sR_ and sL_ to sCounts_
    for (size_t i = 0; i < N_; ++i) {
      for (size_t j = 0; j < N_; ++j) {
        if (j == i) {
          continue;
        }
        ++getSCountsElement(i,getSElement(i,j,SR_MODE));  // Add sR_(i,j)
        ++getSCountsElement(j,getSElement(i,j,SL_MODE));  // Add sL_(j,i)
      }
    }
  }
  void initializeSSObserved() {
    // Add E_ to Phi_
    for (size_t i = 0; i < N_; ++i) {
      for (size_t j = 0; j < N_; ++j) {
        if (j == i) {
          continue;
        }
        Phi_t& PhiEl = getPhiElement(i,j);
        const bool& EEl = getEElement(i,j);
        if (EEl == true) {
          ++PhiEl.get<1>();
        } else {
          ++PhiEl.get<0>();
        }
      }
    }
  }
  
  // Functions to clear sufficient statistics
  void clearSS() {
    clearSSLatent();
    clearSSObserved();
  }
  void clearSSLatent() {
    sCounts_.assign(N_*K_,0);
  }
  void clearSSObserved() {
    Phi_.assign(K_*K_,Phi_t(0,0));
  }
  
  // Gibbs sampler (all latent variables)
  void gs() {
    for (size_t i = 0; i < N_; ++i) {
      for (size_t j = 0; j < N_; ++j) {
        if (j == i) {
          continue;
        }
        gsS(i,j,SR_MODE);  // Gibbs sample E(i,j)'s donor indicator s_{->ij}
        gsS(i,j,SL_MODE);  // Gibbs sample E(i,j)'s receiver indicator s_{<-ij}
      }
    }
  }
  void gsS(size_t i, size_t j, sMode_t sMode) {
    // Samples the element of sR_ or sL_ associated with E(i,j)
    uint16& SEl = getSElement(i,j,sMode);
    bool& EEl = getEElement(i,j);
    size_t entity = (sMode == SR_MODE) ? i : j;
    
    // Decrease sufficient stats
    if (EEl == true) {
      --getPhiElement(i,j).get<1>();
    } else {
      --getPhiElement(i,j).get<0>();
    }
    --getSCountsElement(entity,SEl);
    
    // Sample SEl
    sampleSHelper(i,j,sMode,POSTERIOR);
    
    // Increase sufficient stats
    if (EEl == true) {
      ++getPhiElement(i,j).get<1>();
    } else {
      ++getPhiElement(i,j).get<0>();
    }
    ++getSCountsElement(entity,SEl);
  }
  
  // Generative process functions. These also intialize their associated
  // sufficient statistics.
  //
  // generativeObserved() is NOT appropriate when modNegLinkLL_ is false!
  void generative() {
    generativeLatent();
    generativeObserved();
  }
  void generativeLatent() {
    for (size_t i = 0; i < N_; ++i) {
      for (size_t j = 0; j < N_; ++j) {
        if (j == i) {
          continue;
        }
        generativeS(i,j,SR_MODE); // Sample E(i,j)'s donor indicator s_{->ij}
        generativeS(i,j,SL_MODE); // Sample E(i,j)'s receiver indicator s_{<-ij}
      }
    }
  }
  void generativeS(size_t i, size_t j, sMode_t sMode) {
    // Samples the element of sR_ or sL_ associated with E(i,j)
    // Note: We are conditioning on all sR_, sL_ that have already been sampled
    //       (i.e. prior calls to generativeS())
    uint16& SEl = getSElement(i,j,sMode);
    size_t entity = (sMode == SR_MODE) ? i : j;
    sampleSHelper(i,j,sMode,PRIOR);
    ++getSCountsElement(entity,SEl);
  }
  void generativeObserved() {
    // Requires sCounts_ to be initialized, e.g. by generativeLatent()
    // or initializeSSLatent().
    for (size_t i = 0; i < N_; ++i) {
      for (size_t j = 0; j < N_; ++j) {
        if (j == i) {
          continue;
        }
        generativeE(i,j);
      }
    }
  }
  void generativeE(size_t i, size_t j) {
    // Note: We are conditioning on all edges that have already been sampled
    //       (i.e. prior calls to generativeE())
    bool& EEl = getEElement(i,j);
    if (j == i) {
      EEl = 0;  // Just in case someone calls generativeE(i,i)
    } else {
      Phi_t& PhiEl = getPhiElement(i,j);
      double edgeProb = (PhiEl.get<1>() + lambda1_) /
                        (PhiEl.get<0>() + PhiEl.get<1>() + lambda0_ + lambda1_);
      if (rng_.rand() < edgeProb) {
        EEl = 1;
        ++PhiEl.get<1>();
      } else {
        EEl = 0;
        ++PhiEl.get<0>();
      }
    }
  }
  
  // Marginal log likelihood sampler. Samples every latent variable once,
  // and returns the conditional log likelihood log(P(E|s)) based on the
  // sample. This can be used to estimate the marginal log likelihood
  // log(P(E)) via Monte Carlo integration.
  //
  // Invariant: The sufficient statistics are zero before and after this
  //            function is called.
  double mllSample() {
    generativeLatent();
    initializeSSObserved();
    double ll = computeELogLikelihood();
    clearSS();
    return ll;
  }
  
  // Computes the complete log likelihood log(P(E,s)).
  //
  // Assumes that the sufficient statistics account for ALL E and s.
  double computeCompleteLogLikelihood() const {
    return computeELogLikelihood() + computeSLogPrior();
  }
  
  // Getters for hyperparameter values.
  double getAlpha() const { return alpha_; }
  double getLambda0() const { return lambda0_; }
  double getLambda1() const { return lambda1_; }
  
  // Performs hyperparameter inference using independence chain
  // Metropolis-Hastings.
  //
  // We use proposal distributions equal to the hyperpriors, so
  // their terms cancel out in the acceptance probabilities, leaving
  // just those model terms that depend on the parameter being sampled.
  void mhAlpha() {
    const double ALPHA_PRIOR = 1; // Exponential hyperprior for alpha_
    
    double propAlpha = rng_.randExponential(ALPHA_PRIOR);
    double oldAlpha = alpha_;
    alpha_ = propAlpha;
    double logAcceptProb = computeSLogPrior();
    alpha_ = oldAlpha;
    logAcceptProb -= computeSLogPrior();
    
    if (rng_.rand() < exp(logAcceptProb)) {
      alpha_ = propAlpha; // Accept proposed alpha_
    }
  }
  void mhLambda() {
    const double LAMBDA_PRIOR = 1; // Exponential hyperprior for lambda0/1_
    
    double propLambda0 = rng_.randExponential(LAMBDA_PRIOR);
    double propLambda1 = rng_.randExponential(LAMBDA_PRIOR);
    double oldLambda0 = lambda0_;
    double oldLambda1 = lambda1_;
    lambda0_ = propLambda0;
    lambda1_ = propLambda1;
    double logAcceptProb = computeELogLikelihood();
    lambda0_ = oldLambda0;
    lambda1_ = oldLambda1;
    logAcceptProb -= computeELogLikelihood();
    
    if (rng_.rand() < exp(logAcceptProb)) {
      lambda0_ = propLambda0; // Accept proposed lambda0_ and lambda1_
      lambda1_ = propLambda1;
    }
  }
}; // MmsbSampler

} // namespace ibm

#endif
