use approx::{assert_abs_diff_eq,assert_abs_diff_ne};
use rand::prelude::*;
use rand_distr::Normal;
use rayon::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use ordered_float::OrderedFloat;
use crate::{
    Optimizer,
    Objective,
    simulated_annealing::Particle,
    simulated_annealing::ParticleBuilder,
};

impl<T> Optimizer for ReplicaExchange<T>
    where
        T: Objective + Clone +std::marker::Send,
{
    fn optimize(&mut self, niter: usize) -> (Vec<f64>, f64) {
        self.opt(niter)
    }
}

pub struct ReplicaExchange<T> {
    particles: Vec<Particle<T>>,
    global_best_position: Vec<f64>,
    global_best_score: f64,
    temperature_decay: f64,
    swap_frequency: usize,
}

impl<T: Objective + Clone + std::marker::Send> ReplicaExchange<T> {
    /// Returns particles whose positions are sampled from
    /// a normal distribution defined by the original start position
    /// plus 1.5 * stepsize.
    pub fn new(
            n_particles: usize,
            data: Vec<f64>,
            lower: Vec<f64>,
            upper: Vec<f64>,
            temperature: f64,
            temp_decay: f64,
            stepsize: f64,
            swap_freq: usize,
            object: T,
            accept_from_logit: bool,
    ) -> ReplicaExchange<T>
        where
            T: Objective + Clone,
    {
        // set variance of new particles around data to stepsize^2
        let distr = Normal::new(0.0, stepsize).unwrap();
        //let rng = Arc::new(Mutex::new(Xoshiro256PlusPlus::from_entropy()));
        let mut rng = Xoshiro256PlusPlus::from_entropy();
        //let mut lrng = rng.lock().unwrap();

        // instantiate a Vec
        let mut particle_vec: Vec<Particle<T>> = Vec::new();

        // if doing ReplicaExchange, we'll need geomspace vec of temps
        let t_start = temperature / 3.0;
        let log_start = t_start.log10();
        let t_end = temperature * 3.0;
        let log_end = t_end.log10();

        let dt = (log_end - log_start) / ((n_particles - 1) as f64);
        let mut temps = vec![log_start; n_particles];
        for i in 1..n_particles {
            temps[i] = temps[i - 1] + dt;
        }
        // get temps back from geom space
        let temps: Vec<f64> = temps.iter()
            .map(|x| x.exp())
            .collect();

        // instantiate particles around the actual data
        for i in 0..n_particles {
            //
            let obj_i = object.clone();
            // instantiate the random number generator
            let mut data_vec = data.to_vec();
            // first particle should be right on data
            let temp = temps[i];
            if i == 0 {
                // if this is the first particle, place it directly at data_vec
                let particle = Particle::new(
                    data_vec,
                    lower.to_vec(),
                    upper.to_vec(),
                    obj_i,
                    temp,
                    stepsize,
                    accept_from_logit,
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
                    temp,
                    stepsize,
                    accept_from_logit,
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
        // sort in ascending order of temperature
        particle_vec.sort_unstable_by_key(|a| {
            OrderedFloat(a.temperature)
        });

        ReplicaExchange {
            particles: particle_vec,
            global_best_position: best_pos,
            global_best_score: best_score,
            temperature_decay: temp_decay,
            swap_frequency: swap_freq,
        }
    }

    fn par_iter_mut(&mut self) -> rayon::slice::IterMut<Particle<T>> {
        self.particles.par_iter_mut()
    }

    fn step(&mut self) {
        let t_adj = self.temperature_decay;
        self.par_iter_mut().for_each(|x| {
            x.step(&t_adj);
        });
        self.particles.sort_unstable_by_key(|particle| OrderedFloat(particle.score));
        self.global_best_position = self.particles[0].best_position.to_vec();
        self.global_best_score = self.particles[0].best_score;
    }

    fn exchange(&mut self) {
        let mut iterator = Vec::new();
        let swap_num = self.len() / 2;
        for i in 0..swap_num {
            iterator.push((i*2,i*2+1));
        }
        self.particles.sort_unstable_by_key(
            |particle| OrderedFloat(particle.temperature)
        );
        for swap_idxs in iterator {
            let temp_i = self.particles[swap_idxs.1].temperature;
            let temp_i_plus_one = self.particles[swap_idxs.0].temperature;
            self.particles[swap_idxs.0].temperature = temp_i;
            self.particles[swap_idxs.1].temperature = temp_i_plus_one;
        }
    }

    /// Returns the number of particles in the Particles
    pub fn len(&self) -> usize {self.particles.len()}

    pub fn opt(&mut self, niter: usize) -> (Vec<f64>, f64) {
        for i in 0..niter {
            if i % self.swap_frequency == 0 {
                self.exchange();
            }
            self.step();
        }
        (self.global_best_position.to_vec(), self.global_best_score)
    }
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
    ) -> ReplicaExchange<T>
        where
            T: Objective + Clone + std::marker::Send,
    {

        let data_vec = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let swap_freq = 2;
        let t_decay = 0.01;

        ReplicaExchange::new(
            n_particles,
            data_vec,
            low.to_vec(),
            up.to_vec(),
            temp,
            t_decay,
            *step,
            swap_freq,
            object,
            false,
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
            .set_data(data_vec.to_vec())
            .set_lower(low.to_vec())
            .set_upper(up.to_vec())
            .set_objective(object)
            .set_temperature(temp)
            .set_stepsize(*step)
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

    #[test]
    fn test_temp_swap() {

        let mut particles = set_up_particles(
            6,
            1.0,
            &0.25,
            &vec![-0.5, -0.5, -0.5, -0.5, -0.5],
            &vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            0.5,
            System{},
        );
        let target_temps = vec![
            0.6205672780199154,
            0.7510577348831783,
            0.9089872139693249,
            1.1001254854105642,
            1.3314555640060655,
            1.6114288255590359,
        ];
        for i in 0..6 {
            assert_eq!(particles.particles[i].temperature, target_temps[i]);
        };
        println!("{:?}", particles.particles.iter().map(|x| x.temperature).collect::<Vec<f64>>());
        println!("Scores: {:?}", particles.particles.iter().map(|x| x.score).collect::<Vec<f64>>());

        particles.exchange();
        let swap_target_temps = vec![
            0.7510577348831783,
            0.6205672780199154,
            1.1001254854105642,
            0.9089872139693249,
            1.6114288255590359,
            1.3314555640060655,
        ];
        for i in 0..6 {
            assert_eq!(particles.particles[i].temperature, swap_target_temps[i]);
        }
    }

    #[test]
    fn test_replica_exchange() {
        let obj = System{};
        let step = 0.2;
        let start = vec![0.0, 0.0];
        let low = vec![-5.0, -5.0];
        let up = vec![5.0, 5.0];
        let temp = 10.0;
        let niter = 20000;
        let t_decay = 0.0001;
        let swap_freq = 2;
        let n_particles = 100;

        let mut optimizer = ReplicaExchange::new(
            n_particles,
            start.clone(),
            low.to_vec(),
            up.to_vec(),
            temp,
            t_decay,
            step,
            swap_freq,
            obj,
            false,
        );
        let opt_params = optimizer.optimize(niter);

        let target = vec![1.0,1.0];

        opt_params.0.iter()
            .zip(&start)
            .for_each(|(a,b)| assert_abs_diff_ne!(*a,b));
        opt_params.0.iter()
            .zip(target)
            .for_each(|(a,b)| assert_abs_diff_eq!(*a,b,epsilon=0.03));
        println!("Replica exchange result: {:?}", opt_params);
    }

}
