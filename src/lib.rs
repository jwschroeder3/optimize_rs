use rand::prelude::*;
use std::fmt::Debug;
use rand_distr::{Normal, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use ordered_float::OrderedFloat;
//use std::sync::{Arc, Mutex};
use approx::{assert_abs_diff_eq,assert_abs_diff_ne};
//use rayon::prelude::*;
//use rayon::iter::plumbing::{Producer,bridge};
//use ndarray::prelude::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct System;
    impl Optimizer for System {
        fn objective(&self, param: &Vec<f64>) -> f64 {
            himmelblau(param)
        }
    }

    /// Himmelblau test function (copied directly from argmin-testfunctions
    /// source code then modified slightly)
    ///
    /// Defined as
    ///
    /// `f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2`
    ///
    /// where `x_i \in [-5, 5]`.
    ///
    /// The global minima are at
    ///  * `f(x_1, x_2) = f(3, 2) = 0`.
    ///  * `f(x_1, x_2) = f(-2.805118, 3.131312) = 0`.
    ///  * `f(x_1, x_2) = f(-3.779310, -3.283186) = 0`.
    ///  * `f(x_1, x_2) = f(3.584428, -1.848126) = 0`.
    fn himmelblau(param: &Vec<f64>) -> f64 {
        assert!(param.len() == 2);
        let (x1, x2) = (param[0], param[1]);
        (x1.powi(2) + x2 - 11.0).powi(2)
            + (x1 + x2.powi(2) - 7.0).powi(2)
    }

    /// Multidimensional Rosenbrock test function (copied and slightly modified from
    /// the argmin-testfunctions source)
    ///
    /// Defined as
    ///
    /// `f(x_1, x_2, ..., x_n) = \sum_{i=1}^{n-1} \left[ (a - x_i)^2 + b * (x_{i+1} - x_i^2)^2 \right]`
    ///
    /// where `x_i \in (-\infty, \infty)`. The parameters a and b usually are: `a = 1` and `b = 100`.
    ///
    /// The global minimum is at `f(x_1, x_2, ..., x_n) = f(1, 1, ..., 1) = 0`.
    pub fn rosenbrock(param: &Vec<f64>) -> f64 {
        param.iter()
            .zip(param.iter().skip(1))
            .map(|(&xi, &xi1)| (1.0 - xi).powi(2) + 100.0 * (xi1 - xi.powi(2)).powi(2))
            .sum()
    }

    fn set_up_swarm<'a, T>(
            n_particles: usize,
            temp: f64,
            step: &'a f64,
            low: &'a Vec<f64>,
            up: &'a Vec<f64>,
            init_jitter: f64,
            object: T,
            //objective: fn(&Vec<f64>) -> f64,
            method: &Method,
    ) -> Swarm<T>
        where
            T: Optimizer + Clone,
    {

        let data_vec = vec![0.0, 0.0];

        Swarm::new(
            n_particles,
            data_vec,
            low.to_vec(),
            up.to_vec(),
            temp,
            *step,
            init_jitter,
            object,
            //objective,
            method,
        )
    }

    fn set_up_particle<T>(
            data_vec: &Vec<f64>,
            step: &f64,
            low: &Vec<f64>,
            up: &Vec<f64>,
            object: T,
            //objective: &dyn Fn(&Vec<f64>) -> f64,
            method: &Method
    ) -> Particle<T>
        where
            T: Optimizer,
    {

        let temp = 2.0;
        //let mut rng = thread_rng();
        let particle = Particle::new(
            data_vec.to_vec(),
            low.to_vec(),
            up.to_vec(),
            object,
            //objective,
            temp,
            *step,
            method,
        );
        particle
    }

    #[test]
    fn test_eval() {
        let obj = System{};
        let start_data = vec![0.0, 0.0];
        let step = 0.0;
        let low = vec![-5.0, -5.0];
        let up = vec![5.0, 5.0];
        let particle = set_up_particle(
            &start_data,
            &step,
            &low,
            &up,
            obj,
            //&himmelblau,
            &Method::SimulatedAnnealing,
        );
        let score = particle.evaluate();
        assert_eq!(&score, &170.0);
    }

    #[test]
    fn test_jitter_only() {
        // Here we test that stepsize 0.0 and no velocity do not move particle
        let obj = System{};
        let start_data = vec![0.0, 0.0];
        let step = 0.0;
        let low = vec![-5.0, -5.0];
        let up = vec![5.0, 5.0];
        let mut particle = set_up_particle(
            &start_data,
            &step,
            &low,
            &up,
            obj,
            //&himmelblau,
            &Method::SimulatedAnnealing,
        );

        //let mut rng = Arc::new(Mutex::new(Xoshiro256PlusPlus::from_entropy()));
        particle.perturb();
        // particle should have started at [0.0,0.0], and should not have moved
        // with step of 0.0
        particle.position.iter()
            .zip(&start_data)
            .for_each(|(a,b)| assert_abs_diff_eq!(a,b));

        // Here we test that stepsize 1.0 does move particle
        let obj = System{};
        let step = 1.0;
        let low = vec![-5.0, -5.0];
        let up = vec![5.0, 5.0];
        let mut particle = set_up_particle(
            &start_data,
            &step,
            &low,
            &up,
            obj,
            //&himmelblau,
            &Method::SimulatedAnnealing,
        );
        particle.perturb();
        // particle should end NOT at [1.0,1.0], so the sums should differ
        assert_ne!(
            particle.position.iter().sum::<f64>(),
            start_data.iter().sum::<f64>(),
        );
    }

    #[test]
    fn test_velocity_only() {
        // Here we test that stepsize 0.0 and velocity [1.0, 1.0] moves particle
        // directionally
        let obj = System{};
        let start_data = vec![0.0, 0.0];
        let step = 0.0;
        let inertia = 1.0;
        let local_weight = 0.5;
        let global_weight = 0.5;
        let global_best = vec![4.0, 4.0];
        let low = vec![-5.0, -5.0];
        let up = vec![5.0, 5.0];
        let method = Method::ParticleSwarm;
        let mut particle = set_up_particle(
            &start_data,
            &step,
            &low,
            &up,
            obj,
            //&himmelblau,
            &method,
        );
        // particle velocity is initialized randomly
        let initial_velocity = particle.velocity.to_vec();
        //assert!(particle.velocity.abs_diff_eq(&array![0.0, 0.0], 1e-6));
        particle.set_velocity(
            &inertia,
            &local_weight,
            &global_weight,
            &global_best,
        );
        // particle velocity should have changed
        particle.velocity.iter()
            .zip(&initial_velocity)
            .for_each(|(a,b)| assert_abs_diff_ne!(a,b));
        // move the particle
        println!("Position prior to move: {:?}", &particle.position);
        particle.perturb();
        // particle should have moved in direction of velocity, but magnitude will
        //  be random
        particle.position.iter()
            .zip(&start_data)
            .for_each(|(a,b)| assert_abs_diff_ne!(a,b));
        println!("Position after: {:?}", &particle.position);
    }

    //#[test]
    //fn test_temp_switch() {
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////
    //}

    #[test]
    fn test_annealing() {
        let obj = System{};
        let step = 0.25;
        let start = vec![0.0, 0.0];
        let low = vec![-5.0, -5.0];
        let up = vec![5.0, 5.0];
        let temp = 5.0;
        let niter = 10000;
        let t_adj = 0.0001;

        let opt_params = simulated_annealing(
            start,
            low,
            up,
            temp,
            step,
            niter,
            &t_adj,
            obj,
            //rosenbrock,
            &false,
        );

        println!("Annealing result: {:?}", opt_params);
    }

    #[test]
    fn test_swarming() {

        let obj = System{};
        let step = 0.25;
        let start = vec![0.0, 0.0];
        let low = vec![-5.0, -5.0];
        let up = vec![5.0, 5.0];
        let n_particles = 50;
        let inertia = 0.8;
        let local_weight = 0.2;
        let global_weight = 0.8;
        let initial_jitter = &step * 8.0;

        let niter = 1000;

        let opt_params = particle_swarm(
            start,
            low,
            up,
            n_particles,
            inertia,
            local_weight,
            global_weight,
            initial_jitter,
            niter,
            obj,
            &false,
        );

        println!("Swarm result: {:?}", opt_params);
    }
}

pub enum Method {
    SimulatedAnnealing,
    ParticleSwarm,
    ReplicaExchange,
    BayesianOptimization,
}

#[derive(Debug)]
pub struct Particle<T> {
    object: T,
    position: Vec<f64>,
    prior_position: Vec<f64>,
    best_position: Vec<f64>,
    best_score: f64,
    score: f64,
    lower_bound: Vec<f64>,
    upper_bound: Vec<f64>,
    temperature: f64,
    velocity: Vec<f64>,
    prior_velocity: Vec<f64>,
    stepsize: f64,
    rng: Xoshiro256PlusPlus,
}

impl<T: Optimizer> Particle<T> {
    pub fn new(
        data: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        object: T,
        //objective: &dyn Fn(&Vec<f64>) -> f64,
        temperature: f64,
        stepsize: f64,
        method: &Method,
    ) -> Particle<T>
        where
            T: Optimizer,
    {

        let mut init_rng = Xoshiro256PlusPlus::from_entropy();
        // initialize velocity for each parameter to zero
        let mut v = vec![0.0; data.len()];
        // adjust the starting velocity if we're doing particle swarm
        //let mut rng = thread_rng();
        match method {
            Method::ParticleSwarm => {
                v.iter_mut()
                    .zip(&lower)
                    .zip(&upper)
                    .for_each(|((vel, low), up)| {
                        // draw velocities from uniform dist from +/-(range/40)
                        let init_range = (up - low) / 40.0;
                        let nudge = init_rng.gen_range(-init_range..init_range);
                        *vel = *vel + nudge;
                });
            }
            _ => (),
        }
        let pv = v.to_vec();
        // copy of data to place something into prior_position
        let d = data.to_vec();
        let pr = data.to_vec();

        //let mut rng = Arc::new(Mutex::new(Xoshiro256PlusPlus::from_entropy()));
        let rng = Xoshiro256PlusPlus::from_entropy();
        let mut particle = Particle {
            object: object,
            position: data,
            prior_position: d,
            best_position: pr,
            best_score: f64::INFINITY,
            score: f64::INFINITY,
            lower_bound: lower,
            upper_bound: upper,
            temperature: temperature,
            velocity: v,
            prior_velocity: pv,
            stepsize: stepsize,
            rng: rng,
        };
        particle.score = particle.evaluate();
        particle.best_score = particle.score;
        particle
    }

    /// Gets the score for this Particle
    fn evaluate(
            &self,
    ) -> f64 {
        // the parens are necessary here!
        self.object.objective(&self.position)
    }

    /// Adjusts the position of the Particle
    /// Note that all [Particle]s are instantiated with a velocity of zero.
    /// Therefore, if your optimization algorith does not make use of velocity,
    /// the velocity is never adjusted away from zero, so adding it here does
    /// nothing. If your method *does* use velocity, then it will have adjusted
    /// the velocity such that adding it here has an effect on its position.
    /// Complementary to that, if you want only the velocity to affect particle
    /// position, but no random jitter, set stepsize to 0.0.
    /// Modifies self.position in place.
    fn perturb(&mut self) {

        // before we change the position, set prior position to current position
        // this will enable reversion to prior state if we later reject the move
        self.prior_position = self.position.to_vec();

        // which index will we be nudging?
        let idx = self.choose_param_index();
        // by how far will we nudge?
        let jitter = self.get_jitter();

        // nudge the randomly chosen index by jitter
        self.position[idx] += jitter;
        // add velocity element-wise to position
        self.position.iter_mut() // mutably iterate over each position
            .zip(&self.velocity) // in lockstep with velocity in each dimension
            .zip(&self.lower_bound) // and the lower bounds for each dim
            .zip(&self.upper_bound) // and the upper bounds for each dim
            .for_each(|(((a, b), c), d)| {
                 // update pos, keeping slightly inward of upper and lower bounds
                *a = (*a + b).clamp(*c+f64::EPSILON, *d-f64::EPSILON)
            })
    }

    /// Randomly chooses the index of position to update using jitter
    fn choose_param_index(&mut self) -> usize {
        let die = Uniform::new(0, self.position.len());
        die.sample(&mut self.rng)
    }

    /// Sample once from a normal distribution with a mean of 0.0 and std dev
    /// of self.stepsize.
    fn get_jitter(&mut self) -> f64 {
        // sample from normal distribution one time
        /////////////////////////////////////////////////////
        // Keep this self.rng.lock().unrwrap() for now. the argmin people use it
        // They say it's necessary for thread-safe optims
        /////////////////////////////////////////////////////

        /////////////////////////////////////////////////////
        // I keep wanting to make thist distr a field of Particle, but shouldn't:
        // I want to be able to update stepsize as optimization proceeds
        /////////////////////////////////////////////////////
        let jitter_distr = Normal::new(0.0, self.stepsize).unwrap();
        jitter_distr.sample(&mut self.rng)
    }

    /// Set the velocity of the Particle
    fn set_velocity(&mut self, inertia: &f64,
            local_weight: &f64, global_weight: &f64,
            global_best_position: &Vec<f64>) {
         
        //let mut lrng = rng.lock().unwrap();
        // before we change the velocity, set prior velocity to current velocity
        // this will enable reversion to prior state if we later reject the move
        self.prior_velocity = self.velocity.to_vec();
        // set stochastic element of weights applied to local and global best pos
        // self.rng.gen samples from [0.0, 1.0)
        let r_arr: [f64; 2] = self.rng.gen();
        // set the new velocity
        self.velocity.iter_mut() // mutably iterate over current velocity
            .zip(&self.best_position) // in lockstep with this Particle's best position
            .zip(global_best_position) // and the global best position
            .zip(&self.position) // and the current position
            .zip(&self.lower_bound) // and the lower bound
            .zip(&self.upper_bound) // and the upper bound
            // a=vel, b=local_best, c=swarm_best, d=pos, e=lower_bound, f=upper_bound
            .for_each(|(((((a, b), c), d), e), f)| {
                //let range = f - e;
                let term1 = inertia * *a;
                // attraction to the particle's own best gets stronger with distance
                let term2 = local_weight * r_arr[0] * (b - d);
                // attraction to the swarms's best gets stronger with distance
                let term3 = global_weight * r_arr[1] * (c - d);
                // repulsion from lower bound defined by squared distance to lower bound
                //let term4 = -(range / 100.0) / ((e - d) * (e - d));
                // repulsion from upper bound defined by squared distance to upper bound
                //let term5 = -(range / 100.0) / ((f - d) * (f - d));
                *a = term1 + term2 + term3// + term4 + term5
            })
    }

    fn step(
            &mut self,
            //objective: &dyn Fn(&Vec<f64>) -> f64,
            //object: &T,
            inertia: &f64,
            local_weight: &f64,
            global_weight: &f64,
            global_best_position: &Vec<f64>,
            t_adj: &f64,
            method: &Method,
            accept_from_logit: &bool,
    )
        where
            T: Optimizer,
    {
        // set the new velocity
        self.set_velocity(
            inertia,
            local_weight,
            global_weight,
            global_best_position,
        );
        // move the particle. Takes into account stepsize for jitter in a single
        //  dimension, and velocity over all dimensions.
        self.perturb();

        let score = self.evaluate(); //, rec_db, kmer, max_count, alpha);

        match method {
            // if we are doing particle swarm, just update and move on
            Method::ParticleSwarm => self.update_scores(&score),
            // If we are doing simulated annealing,
            //  determine whether we accept the move.
            //  if we reject, revert to prior state and perturb again.
            Method::SimulatedAnnealing => {
                if !self.accept(&score, accept_from_logit) {
                    self.revert();
                } else {
                    // Update prior score [and possibly the best score] if we accepted
                    self.update_scores(&score);
                }
            }
            Method::ReplicaExchange => {
                if !self.accept(&score, accept_from_logit) {
                    self.revert();
                } else {
                    // Update prior score [and possibly the best score] if we accepted
                    self.update_scores(&score);
                }
            }
            Method::BayesianOptimization => todo!()
        }
        // adjust the temperature downward
        ////////////////////////////////////////////////////
        // NOTE: this could be included in the match control flow above
        // I have to check if temp adjustment is used in replica exchange,
        // or whether we just let the temps remain constant.
        ////////////////////////////////////////////////////
        self.adjust_temp(t_adj);
    }

    /// Update score fields after accepting a move
    fn update_scores(&mut self, score: &f64) {
        self.score = *score;
        // if this was our best-ever score, update best_score and best_position
        if *score < self.best_score {
            self.best_score = *score;
            self.best_position = self.position.to_vec();
        }
    }
    
    /// Revert current position and velocity to prior values
    fn revert(&mut self) {
        self.position = self.prior_position.to_vec();
        self.velocity = self.prior_velocity.to_vec();
    }
    
    /// Determine whether to accept the new position, or to go back to prior
    /// position and try again. If score is greater that prior score,
    /// return true. If the score is less than prior score, determine whether
    /// to return true probabilistically using the following function:
    ///
    /// `exp(-(score - prior_score)/T)`
    ///
    /// where T is temperature.
    fn accept(&mut self, score: &f64, accept_from_logit: &bool) -> bool {
        if *accept_from_logit {
            // clamp score to just barely above -1.0, up to just barely below 0.0
            // this avoids runtime error in logit
            let clamp_score = score.clamp(-(1.0-f64::EPSILON), -f64::EPSILON);
            // take opposite of scores, since they're opposite of AMI
            let diff = logit(&-clamp_score)
                - logit(&-self.score.clamp(-(1.0-f64::EPSILON), -f64::EPSILON));
            // testing if diff >= 0.0 here, and NOT <= 0.0, since we're doing
            //   that thing of when we compare the logit of the opposite of
            //   the scores. Therefore, a greater score than our previous
            //   score is desireable.
            if diff >= 0.0 {
                true
            // if score > last score, decide probabilistically whether to accept
            } else {
                // this is the function used by scipy.optimize.basinhopping for P(accept)
                // the only difference here is that, again, because we're really maximizing
                // a score, we just use diff, and not -diff.
                let accept_prob = (diff/self.temperature).exp();
                if accept_prob > self.rng.gen() {
                    true
                }
                else {
                    false
                }
            }
        } else {
            // compare this score to prior score
            let diff = score - self.score;
            if diff <= 0.0 {
                true
            } else {
                // this is the function used by scipy.optimize.basinhopping for P(accept)
                let accept_prob = (-diff/self.temperature).exp();
                if accept_prob > self.rng.gen() {
                    true
                }
                else {
                    false
                }
            }

        }
    }

    /// Adjusts the temperature of the Particle
    ///
    /// Defined as
    ///
    /// ` T_{i+1} = T_i * (1.0 - t_{adj})`
    ///
    /// # Arguments
    ///
    /// * `t_adj` - fractional amount by which to decrease the temperature of
    ///    the particle. For instance, if current temp is 1.0 and t_adj is 0.2,
    ///    the new temperature will be 1.0 * (1.0 - 0.2) = 0.8
    fn adjust_temp(&mut self, t_adj: &f64) {
        self.temperature *= (1.0 - t_adj)
    }
}

/// A trait to enable optimization of a set of parameters for
/// any struct
pub trait Optimizer {
    fn objective(&self, theta: &Vec<f64>) -> f64;
    //fn evaluate(&self, objective: &dyn Fn(&Vec<f64>) -> f64) -> f64;
    //fn perturb(&mut self);
    //fn choose_param_index(&mut self) -> usize;
    //fn get_jitter(&mut self) -> f64;
    //fn set_velocity(&mut self, inertia: &f64,
    //        local_weight: &f64, global_weight: &f64,
    //        global_best_position: &Vec<f64>);
    //fn step(
    //        &mut self,
    //        objective: &dyn Fn(&Vec<f64>) -> f64,
    //        inertia: &f64,
    //        local_weight: &f64,
    //        global_weight: &f64,
    //        global_best_position: &Vec<f64>,
    //        t_adj: &f64,
    //        method: &Method,
    //        accept_from_logit: &bool,
    //);
    //fn update_scores(&mut self, score: &f64);
    //fn revert(&mut self);
    //fn accept(&mut self, score: &f64, accept_from_logit: &bool) -> bool;
    //fn adjust_temp(&mut self, t_adj: &f64);
}

///// A trait to organize Particles into a Swarm
//pub trait Swarm {
//    fn new(particles: Vec<Box<T>>, objective: Fn(&Vec<f65>) -> f64) -> Self
//}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
// workaround for eliminating redundant code for implementing trait for structs with different fields
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
//macro_rules! add_optimizer_impl {
//    ($struct_name:ident { $($field_name:ident : $field_type:ty),* }) => {
//        #[derive(Debug)]
//        pub struct $struct_name {
//            $($field_name : $field_type,)*
//            position: Vec<f64>,
//            prior_position: Vec<f64>,
//            best_position: Vec<f64>,
//            best_score: f64,
//            score: f64,
//            lower_bound: Vec<f64>,
//            upper_bound: Vec<f64>,
//            temperature: f64,
//            velocity: Vec<f64>,
//            prior_velocity: Vec<f64>,
//            stepsize: f64,
//            rng: Xoshiro256PlusPlus,
//        }
//
//        impl Optimizer for $struct_name {
//        }
//}

// The idea here is to implement Swarm for Swarm.
//pub struct Swarm<T> {
pub struct Swarm<T> {
    particles: Vec<Particle<T>>,
    global_best_position: Vec<f64>,
    global_best_score: f64,
    //objective: fn(&Vec<f64>) -> f64,
    //object: T,
}

//type Data = Particle;
//
//impl<T> Debug for Swarm<T> {
//    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//        self.particles.fmt(f)
//    }
//}
//
//impl<'a, T> IntoParallelIterator for &'a Swarm<T> {
//    type Iter = ParDataIter<'a>;
//    type Item = &'a Data;
//
//    fn into_par_iter(self) -> Self::Iter {
//        ParDataIter { data: &self.particles }
//    }
//}
//
//impl<'a, T: std::marker::Send> IntoParallelIterator for &'a mut Swarm<T> {
//    type Iter = ParDataIterMut<'a, T>;
//    type Item = &'a mut Data;
//
//    fn into_par_iter(self) -> Self::Iter {
//        ParDataIterMut { data: self }
//    }
//}
//
//pub struct ParDataIter<'a> {
//    data: &'a [Data],
//}
//
//pub struct ParDataIterMut<'a, T> {
//    data: &'a mut Swarm<T>,
//}
//
//impl<'a> ParallelIterator for ParDataIter<'a> {
//    type Item = &'a Data;
//
//    fn drive_unindexed<C>(self, consumer: C) -> C::Result
//        where
//        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
//    {
//        bridge(self, consumer)
//    }
//
//    fn opt_len(&self) -> Option<usize> {
//        Some(<Self as IndexedParallelIterator>::len(self))
//    }
//}
//
//impl<'a, T: std::marker::Send> ParallelIterator for ParDataIterMut<'a, T> {
//    type Item = &'a mut Data;
//    fn drive_unindexed<C>(self, consumer: C) -> C::Result
//        where
//        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
//    {
//        bridge(self, consumer)
//    }
//
//    fn opt_len(&self) -> Option<usize> {
//        Some(<Self as IndexedParallelIterator>::len(self))
//    }
//}
//
//impl<'a> IndexedParallelIterator for ParDataIter<'a> {
//    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
//        self,
//        callback: CB,
//    ) -> CB::Output {
//        let data_producer = DataProducer::from(self);
//        callback.callback(data_producer)
//    }
//
//    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
//        bridge(self, consumer)
//    }
//
//    fn len(&self) -> usize {
//        self.data.len()
//    }
//}
//
//impl<'a, T: std::marker::Send> IndexedParallelIterator for ParDataIterMut<'a, T> {
//    fn with_producer<CB: rayon::iter::plumbing::ProducerCallback<Self::Item>>(
//        self,
//        callback: CB,
//    ) -> CB::Output {
//        let producer = DataProducerMut::from(self);
//        callback.callback(producer)
//    }
//
//    fn drive<C: rayon::iter::plumbing::Consumer<Self::Item>>(self, consumer: C) -> C::Result {
//        bridge(self, consumer)
//    }
//
//    fn len(&self) -> usize {
//        self.data.particles.len()
//    }
//}
//
//pub struct DataProducer<'a> {
//    data_slice: &'a [Data],
//}
//
//pub struct DataProducerMut<'a> {
//    data_slice: &'a mut [Data],
//}
//
//impl<'a> From<&'a mut [Data]> for DataProducerMut<'a> {
//    fn from(data_slice: &'a mut [Data]) -> Self {
//        Self { data_slice }
//    }
//}
//
//impl<'a> From<ParDataIter<'a>> for DataProducer<'a> {
//    fn from(iterator: ParDataIter<'a>) -> Self {
//        Self {
//            data_slice: &iterator.data,
//        }
//    }
//}
//
//impl<'a, T> From<ParDataIterMut<'a, T>> for DataProducerMut<'a> {
//    fn from(iterator: ParDataIterMut<'a, T>) -> Self {
//        Self {
//            data_slice: &mut iterator.data.particles,
//        }
//    }
//}
//
//impl<'a> Producer for DataProducer<'a> {
//    type Item = &'a Data;
//    type IntoIter = std::slice::Iter<'a, Data>;
//
//    fn into_iter(self) -> Self::IntoIter {
//        self.data_slice.iter()
//    }
//
//    fn split_at(self, index: usize) -> (Self, Self) {
//        let (left, right) = self.data_slice.split_at(index);
//        (
//            DataProducer { data_slice: left },
//            DataProducer { data_slice: right },
//        )
//    }
//}
//
//impl<'a> Producer for DataProducerMut<'a> {
//    type Item = &'a mut Data;
//    type IntoIter = std::slice::IterMut<'a, Data>;
//
//    fn into_iter(self) -> Self::IntoIter {
//        self.data_slice.iter_mut()
//    }
//
//    fn split_at(self, index: usize) -> (Self, Self) {
//        let (left, right) = self.data_slice.split_at_mut(index);
//        (Self::from(left), Self::from(right))
//    }
//}

impl<T: Optimizer + Clone> Swarm<T> {
    /// Returns a Swarm of particles whose positions are sampled from
    /// a normal distribution defined by the original start position
    /// plus 1.5 * stepsize.
    pub fn new(
            n_particles: usize,
            data: Vec<f64>,
            lower: Vec<f64>,
            upper: Vec<f64>,
            temperature: f64,
            stepsize: f64,
            initial_jitter: f64,
            object: T,
            //objective: fn(&Vec<f64>) -> f64,
            method: &Method,
    ) -> Swarm<T>
        where
            T: Optimizer + Clone,
    {
        // set variance of new particles around data to initial_jitter^2
        let distr = Normal::new(0.0, initial_jitter).unwrap();
        //let rng = Arc::new(Mutex::new(Xoshiro256PlusPlus::from_entropy()));
        let mut rng = Xoshiro256PlusPlus::from_entropy();
        //let mut lrng = rng.lock().unwrap();

        // instantiate a Vec
        let mut particle_vec: Vec<Particle<T>> = Vec::new();
        // instantiate particles around the actual data
        for i in 0..n_particles {
            //
            let obj_i = object.clone();
            // instantiate the random number generator
            let mut data_vec = data.to_vec();
            let mut temp = 0.0;
            // first particle should be right on data
            match method {
                Method::SimulatedAnnealing => {
                    temp = temperature;
                }
                Method::ReplicaExchange => {
                    let temp_distr = Uniform::new(temperature / 3.0, temperature*3.0);
                    temp = temp_distr.sample(&mut rng);
                }
                Method::ParticleSwarm => {
                    temp = 0.0;
                }
                Method::BayesianOptimization => todo!()
            }
            if i == 0 {
                // if this is the first particle, place it directly at data_vec
                let particle = Particle::new(
                    data_vec,
                    lower.to_vec(),
                    upper.to_vec(),
                    obj_i,
                    //&objective,
                    temp,
                    stepsize,
                    method,
                );
                particle_vec.push(particle);
            } else {
                let mut rng = Xoshiro256PlusPlus::from_entropy();
                data_vec.iter_mut()
                    .enumerate()
                    .for_each(|(i,a)| {
                        // set new particle's data to data + sample, clamp between bounds
                        *a = *a + distr
                            .sample(&mut rng)
                            .clamp(lower[i],upper[i]);
                    });
                let particle = Particle::new(
                    data_vec,
                    lower.to_vec(),
                    upper.to_vec(),
                    obj_i,
                    //&objective,
                    temp,
                    stepsize,
                    method,
                );
                particle_vec.push(particle);
            }
        }
        // sort in ascending order of score
        particle_vec.sort_unstable_by_key(|a| {
            OrderedFloat(a.score)
        });
        // lowest score is best, so take first one's position and score
        let best_pos = particle_vec[0].best_position.to_vec();
        let best_score = particle_vec[0].best_score;
        Swarm {
            particles: particle_vec,
            global_best_position: best_pos,
            global_best_score: best_score,
            //objective: objective,
            //object: *object,
        }
    }

    fn step(
            &mut self,
            inertia: &f64,
            local_weight: &f64,
            global_weight: &f64,
            t_adj: &f64,
            method: &Method,
            accept_from_logit: &bool,
    ) {
        let global_best = self.global_best_position.to_vec();
        //let obj = self.objective;
        //let obj = &self.object;
        for x in self.particles.iter_mut() {
        //self.par_iter_mut().for_each(|x| {
            x.step(
                //obj,
                &inertia,
                &local_weight,
                &global_weight,
                &global_best,
                t_adj,
                method,
                accept_from_logit,
            );
        }
        //});
        self.particles.sort_unstable_by_key(|particle| OrderedFloat(particle.score));
        self.global_best_position = self.particles[0].best_position.to_vec();
        self.global_best_score = self.particles[0].best_score;
    }

    pub fn exchange(&mut self, odds: bool) {
        let mut iterator = Vec::new();
        if odds {
            let swap_num = self.len() / 2;
            for i in 0..swap_num {
                iterator.push((i*2,i*2+1));
            }
        } else {
            let swap_num = self.len() / 2 - 1;
            for i in 0..swap_num {
                iterator.push((i*2,i*2+1));
            }
        }
        self.particles.sort_unstable_by_key(|particle| OrderedFloat(particle.temperature));
        for swap_idxs in iterator {
            let temp_i = self.particles[swap_idxs.1].temperature;
            let temp_i_plus_one = self.particles[swap_idxs.0].temperature;
            self.particles[swap_idxs.0].temperature = temp_i;
            self.particles[swap_idxs.1].temperature = temp_i_plus_one;
        }
    }

    /// Returns the number of particles in the Swarm
    pub fn len(&self) -> usize {self.particles.len()}
}

///////////////////////////////////////////////////
// I may use this in acceptance criterion /////////
///////////////////////////////////////////////////
pub fn logit(p: &f64) -> f64 {
    (p / (1.0-p)).ln()
}

//add_optimizer_impl!(Particle{});

pub fn simulated_annealing<T>(
        params: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        temp: f64,
        step: f64,
        niter: usize,
        t_adj: &f64,
        object: T,
        //objective: fn(&Vec<f64>) -> f64,
        accept_from_logit: &bool,
) -> (Vec<f64>, f64)
    where
        T: Optimizer + Clone,
{

    // set inertia, local_weight, and global_weight to 0.0 to turn off velocities,
    // thus leaving only the jitter to affect particle position
    let inertia = 0.0;
    let local_weight = 0.0;
    let global_weight = 0.0;

    let mut swarm = Swarm::new(
        1, // n_particles is always 1 for simulated annealing
        params, // Vec<f64>
        lower,
        upper,
        temp,
        step,
        0.0, // initial_jitter is 0.0 to place particle exactly at data
        object,
        //objective,
        &Method::SimulatedAnnealing,
    );

    for _ in 0..niter {
        swarm.step(
            &inertia,
            &local_weight,
            &global_weight,
            &t_adj,
            &Method::SimulatedAnnealing,
            accept_from_logit,
        );
    }
    (swarm.global_best_position.to_vec(), swarm.global_best_score)
}

pub fn particle_swarm<T>(
        params: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        n_particles: usize,
        inertia: f64,
        local_weight: f64,
        global_weight: f64,
        initial_jitter: f64,
        niter: usize,
        object: T,
        accept_from_logit: &bool,
) -> (Vec<f64>, f64)
    where
        T: Optimizer + Clone,
{
    // turn off jitter, leaving only velocity to affect position
    let step = 0.0;
    let temp = 0.0;
    let t_adj = 0.0;

    let mut swarm = Swarm::new(
        n_particles,
        params,
        lower,
        upper,
        temp,
        step,
        initial_jitter,
        //objective,
        object,
        &Method::ParticleSwarm,
    );

    //let mut rng = thread_rng();

    for _ in 0..niter {
        swarm.step(
            &inertia,
            &local_weight,
            &global_weight,
            &t_adj,
            &Method::ParticleSwarm,
            accept_from_logit,
        );
    }
    (swarm.global_best_position.to_vec(), swarm.global_best_score)
}


