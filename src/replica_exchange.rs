use approx::assert_abs_diff_eq;
use crate::{
    Optimizer,
    Objective,
};

impl<T> Optimizer<T> for ReplicaExchange {
    fn step(&mut self) {}
}

pub struct ReplicaExchange<T> {
    particles: Vec<Particle<T>>,
    global_best_position: Vec<f64>,
    global_best_score: f64,
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
    stepsize: f64,
    rng: Xoshiro256PlusPlus,
}

impl<T: Objective + std::marker::Send> Particle<T> {
    pub fn new(
        data: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        object: T,
        temperature: f64,
        stepsize: f64,
    ) -> Particle<T>
        where
            T: Objective + std::marker::Send,
    {

        let mut init_rng = Xoshiro256PlusPlus::from_entropy();
        // copy of data to place something into prior_position
        let d = data.to_vec();
        let pr = data.to_vec();

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
    }

    /// Randomly chooses the index of position to update using jitter
    fn choose_param_index(&mut self) -> usize {
        let die = Uniform::new(0, self.position.len());
        die.sample(&mut self.rng)
    }

    /// Sample once from a normal distribution with a mean of 0.0 and std dev
    /// of self.stepsize.
    fn get_jitter(&mut self) -> f64 {
        /////////////////////////////////////////////////////
        // I keep wanting to make thist distr a field of Particle, but shouldn't:
        // I want to be able to update stepsize as optimization proceeds
        /////////////////////////////////////////////////////
        let jitter_distr = Normal::new(0.0, self.stepsize).unwrap();
        jitter_distr.sample(&mut self.rng)
    }

    fn step(
            &mut self,
            global_best_position: &Vec<f64>,
            t_adj: &f64,
            accept_from_logit: &bool,
    )
        where
            T: Objective + std::marker::Send,
    {
        // move the particle. 
        self.perturb();
        let score = self.evaluate(); //, rec_db, kmer, max_count, alpha);

        if !self.accept(&score, accept_from_logit) {
            self.revert();
        } else {
            // Update prior score [and possibly the best score] if we accepted
            self.update_scores(&score);
        }
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
        self.temperature *= 1.0 - t_adj
    }

    /// Adjusts that stepsize of the Particle
    ///
    /// Definaed as
    ///
    /// ` S_{i+1} = S_i * (1.0 - s_{adj})`
    ///
    /// # Arguments
    ///
    /// * `s_adj` - fractional amount by which to decrease the stepsize of the
    ///     particle.
    fn adjust_stepsize(&mut self, s_adj: &f64) {
        self.stepsize *= 1.0 - s_adj
    }
}

pub fn replica_exchange<T>(
        params: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        n_particles: usize,
        temp: f64,
        step: f64,
        initial_jitter: f64,
        niter: usize,
        t_adj: &f64,
        swap_freq: &usize,
        object: T,
        accept_from_logit: &bool,
) -> (Vec<f64>, f64)
    where
        T: Objective + Clone + std::marker::Send,
{

    // set inertia, local_weight, and global_weight to 0.0 to turn off velocities,
    // thus leaving only the jitter to affect particle position
    let inertia = 0.0;
    let local_weight = 0.0;
    let global_weight = 0.0;

    let mut particles = Particles::new(
        n_particles,
        params,
        lower,
        upper,
        temp,
        step,
        initial_jitter,
        object,
        &Method::ReplicaExchange,
    );

    for i in 0..niter {
        if i % swap_freq == 0 {
            particles.exchange(true);
        }
        particles.step(
            &inertia,
            &0.0,
            &0.0,
            &0.0,
            &local_weight,
            &global_weight,
            &t_adj,
            accept_from_logit,
        );
    }
    (particles.global_best_position.to_vec(), particles.global_best_score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct System;
    impl Objective for System {
        fn objective(&self, param: &Vec<f64>) -> f64 {
            rosenbrock(param)
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

    fn set_up_particles<'a, T>(
            n_particles: usize,
            temp: f64,
            step: &'a f64,
            low: &'a Vec<f64>,
            up: &'a Vec<f64>,
            init_jitter: f64,
            object: T,
            method: &Method,
    ) -> Particles<T>
        where
            T: Objective + Clone + std::marker::Send,
    {

        let data_vec = vec![0.0, 0.0, 0.0, 0.0, 0.0];

        Particles::new(
            n_particles,
            data_vec,
            low.to_vec(),
            up.to_vec(),
            temp,
            *step,
            init_jitter,
            object,
            method,
        )
    }

    fn set_up_particle<T>(
            data_vec: &Vec<f64>,
            step: &f64,
            low: &Vec<f64>,
            up: &Vec<f64>,
            object: T,
            method: &Method
    ) -> Particle<T>
        where
            T: Objective + std::marker::Send + Clone,
    {

        let temp = 2.0;
        let particle = ParticleBuilder::new()
            .set_data(data_vec.to_vec()).unwrap()
            .set_lower(low.to_vec()).unwrap()
            .set_upper(up.to_vec()).unwrap()
            .set_objective(object)
            .set_temperature(temp).unwrap()
            .set_stepsize(*step).unwrap()
            .set_method(*method)
            .build().unwrap();
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
            &Method::SimulatedAnnealing,
        );
        let score = particle.evaluate();
        // for himmelblau
        //assert_eq!(&score, &170.0);
        // for rosenbrock
        assert_eq!(&score, &1.0);
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
            &Method::SimulatedAnnealing,
        );
        particle.perturb();
        // particle should end NOT at [1.0,1.0], so the sums should differ
        assert_ne!(
            particle.position.iter().sum::<f64>(),
            start_data.iter().sum::<f64>(),
        );
    }

    //#[test]
    //fn test_velocity_only() {
    //    // Here we test that stepsize 0.0 and velocity [1.0, 1.0] moves particle
    //    // directionally
    //    let obj = System{};
    //    let start_data = vec![0.0, 0.0];
    //    let step = 0.0;
    //    let inertia = 1.0;
    //    let local_weight = 0.5;
    //    let global_weight = 0.5;
    //    let global_best = vec![4.0, 4.0];
    //    let low = vec![-5.0, -5.0];
    //    let up = vec![5.0, 5.0];
    //    let method = Method::ParticleSwarm;
    //    let mut particle = set_up_particle(
    //        &start_data,
    //        &step,
    //        &low,
    //        &up,
    //        obj,
    //        //&himmelblau,
    //        &method,
    //    );
    //    // particle velocity is initialized randomly
    //    let initial_velocity = particle.velocity.to_vec();
    //    //assert!(particle.velocity.abs_diff_eq(&array![0.0, 0.0], 1e-6));
    //    particle.set_velocity(
    //        &inertia,
    //        &local_weight,
    //        &global_weight,
    //        &global_best,
    //        &method,
    //    );
    //    // particle velocity should have changed
    //    particle.velocity.iter()
    //        .zip(&initial_velocity)
    //        .for_each(|(a,b)| assert_abs_diff_ne!(a,b));
    //    // move the particle
    //    println!("Position prior to move: {:?}", &particle.position);
    //    particle.perturb(&method);
    //    // particle should have moved in direction of velocity, but magnitude will
    //    //  be random
    //    particle.position.iter()
    //        .zip(&start_data)
    //        .for_each(|(a,b)| assert_abs_diff_ne!(a,b));
    //    println!("Position after: {:?}", &particle.position);
    //}

    //#[test]
    //fn test_temp_swap() {
    //    let mut particles = set_up_particles(
    //        6,
    //        1.0,
    //        &0.25,
    //        &vec![-0.5, -0.5, -0.5, -0.5, -0.5],
    //        &vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    //        0.5,
    //        System{},
    //        &Method::ReplicaExchange,
    //    );
    //    for i in 0..6 {
    //        println!("{:?}", particles.particles[i].temperature);
    //    }
    //    particles.exchange(false);
    //    println!("");
    //    for i in 0..6 {
    //        println!("{:?}", particles.particles[i].temperature);
    //    }
    //}

    //#[test]
    //fn test_replica_exchange() {
    //    let obj = System{};
    //    let step = 0.2;
    //    let start = vec![0.0, 0.0];
    //    let low = vec![-5.0, -5.0];
    //    let up = vec![5.0, 5.0];
    //    let temp = 10.0;
    //    let niter = 20000;
    //    let t_adj = 0.0001;
    //    let initial_jitter = &step * 8.0;
    //    let swap_freq = 5;
    //    let n_particles = 50;

    //    let opt_params = replica_exchange(
    //        start.clone(),
    //        low,
    //        up,
    //        n_particles,
    //        temp,
    //        step,
    //        initial_jitter,
    //        niter,
    //        &t_adj,
    //        &swap_freq,
    //        obj,
    //        &false,
    //    );

    //    let target = vec![1.0,1.0];

    //    opt_params.0.iter()
    //        .zip(&start)
    //        .for_each(|(a,b)| assert_abs_diff_ne!(*a,b));
    //    opt_params.0.iter()
    //        .zip(target)
    //        .for_each(|(a,b)| assert_abs_diff_eq!(*a,b,epsilon=0.03));
    //    println!("Replica exchange result: {:?}", opt_params);
    //}

    //#[test]
    //fn test_annealing() {
    //    let obj = System{};
    //    let step = 0.2;
    //    let start = vec![0.0, 0.0];
    //    let low = vec![-5.0, -5.0];
    //    let up = vec![5.0, 5.0];
    //    let temp = 10.0;
    //    let niter = 20000;
    //    let t_adj = 0.0001;

    //    let opt_params = simulated_annealing(
    //        start.clone(),
    //        low,
    //        up,
    //        temp,
    //        step,
    //        niter,
    //        &t_adj,
    //        obj,
    //        //rosenbrock,
    //        &false,
    //    );

    //    let target = vec![1.0,1.0];

    //    opt_params.0.iter()
    //        .zip(&start)
    //        .for_each(|(a,b)| assert_abs_diff_ne!(*a,b));
    //    opt_params.0.iter()
    //        .zip(target)
    //        .for_each(|(a,b)| assert_abs_diff_eq!(*a,b,epsilon=0.03));
    //    println!("Annealing result: {:?}", opt_params);
    //}

    //#[test]
    //fn test_swarming() {

    //    let obj = System{};
    //    let step = 0.25;
    //    let start = vec![0.0, 0.0];
    //    let low = vec![-5.0, -5.0];
    //    let up = vec![5.0, 5.0];
    //    let n_particles = 50;
    //    let inertia = 0.8;
    //    let local_weight = 0.2;
    //    let global_weight = 0.8;
    //    let initial_jitter = &step * 8.0;

    //    let niter = 1000;

    //    let opt_params = particle_swarm(
    //        start.clone(),
    //        low,
    //        up,
    //        n_particles,
    //        inertia,
    //        local_weight,
    //        global_weight,
    //        initial_jitter,
    //        niter,
    //        obj,
    //        &false,
    //    );

    //    let target = vec![1.0,1.0];

    //    opt_params.0.iter()
    //        .zip(&start)
    //        .for_each(|(a,b)| assert_abs_diff_ne!(*a,b));
    //    opt_params.0.iter()
    //        .zip(target)
    //        .for_each(|(a,b)| assert_abs_diff_eq!(*a,b));

    //    println!("Swarm result: {:?}", opt_params);
    //}

    //#[test]
    //fn test_pidao() {

    //    let obj = System{};
    //    let step = 0.25;
    //    let start = vec![0.0, 0.0];
    //    let low = vec![-5.0, -5.0];
    //    let up = vec![5.0, 5.0];

    //    let niter = 1000;

    //    let opt_params = pidao(
    //        start.clone(),
    //        low,
    //        up,
    //        step,
    //        niter,
    //        obj,
    //        &false,
    //    );

    //    let target = vec![1.0,1.0];

    //    opt_params.0.iter()
    //        .zip(&start)
    //        .for_each(|(a,b)| assert_abs_diff_ne!(*a,b));
    //    opt_params.0.iter()
    //        .zip(target)
    //        .for_each(|(a,b)| assert_abs_diff_eq!(*a,b));

    //    println!("PIDAO result: {:?}", opt_params);
    //}

}
