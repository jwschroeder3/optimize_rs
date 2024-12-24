use approx::assert_abs_diff_eq;
use crate::{
    Optimizer,
    Objective,
    Method,
    Particle,
    ParticleBuilder,
};

pub struct BayesianOptimization {
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
            &Method::BayesianOptimization,
        )
    }

    fn set_up_particle<T>(
            data_vec: &Vec<f64>,
            step: &f64,
            low: &Vec<f64>,
            up: &Vec<f64>,
            object: T,
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
            .set_method(Method::BayesianOptimization)
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
    //    let mut particle = set_up_particle(
    //        &start_data,
    //        &step,
    //        &low,
    //        &up,
    //        obj,
    //        //&himmelblau,
    //    );
    //    // particle velocity is initialized randomly
    //    let initial_velocity = particle.velocity.to_vec();
    //    //assert!(particle.velocity.abs_diff_eq(&array![0.0, 0.0], 1e-6));
    //    particle.set_velocity(
    //        &inertia,
    //        &local_weight,
    //        &global_weight,
    //        &global_best,
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
