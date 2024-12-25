use approx::{assert_abs_diff_eq,assert_abs_diff_ne};
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use rand::prelude::*;
use rand_distr::Normal;
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::{
    Objective,
    Optimizer,
    Method,
    IncompleteParticleBuild,
    InvalidParticleBuild,
};

pub struct ParticleBuilder<T> {
    object: Option<T>,
    position: Option<Vec<f64>>,
    score: Option<f64>,
    lower_bound: Option<Vec<f64>>,
    upper_bound: Option<Vec<f64>>,
    velocity: Option<Vec<f64>>,
    velocity_decay: Option<f64>,
    inertia: Option<f64>,
    initial_jitter: Option<f64>,
    local_weight: Option<f64>,
    global_weight: Option<f64>,
    accept_from_logit: Option<bool>,
    rng: Xoshiro256PlusPlus,
}

impl<T: Objective + std::marker::Send + Clone> ParticleBuilder<T> {
    fn new() -> Self
        where
            T: Objective + Clone + std::marker::Send,
    {
        let rng = Xoshiro256PlusPlus::from_entropy();
        Self{
            object: None,
            position: None,
            score: None,
            lower_bound: None,
            upper_bound: None,
            velocity: None,
            velocity_decay: None,
            inertia: None,
            initial_jitter: None,
            local_weight: None,
            global_weight: None,
            accept_from_logit: None,
            rng: rng,
        }
    }

    fn set_data(&mut self, data: Vec<f64>) -> &mut Self {
        self.position = Some(data);
        self
    }

    fn set_lower(&mut self, lower: Vec<f64>) -> &mut Self {
        self.lower_bound = Some(lower);
        self
    }

    fn set_upper(&mut self, upper: Vec<f64>) -> &mut Self {
        self.upper_bound = Some(upper);
        self
    }

    fn set_objective(&mut self, objective: T) ->
        &mut Self
            where
        T: Objective + std::marker::Send
    {
        self.object = Some(objective);
        self
    }

    fn set_inertia(&mut self, inertia: f64) -> &mut Self {
        self.inertia = Some(inertia);
        self
    }

    fn set_local_weight(&mut self, weight: f64) -> &mut Self {
        self.local_weight = Some(weight);
        self
    }

    fn set_global_weight(&mut self, weight: f64) -> &mut Self {
        self.global_weight = Some(weight);
        self
    }

    fn set_accept_from_logit(&mut self, accept_from_logit: bool) -> &mut Self {
        self.accept_from_logit = Some(accept_from_logit);
        self
    }

    fn set_decay(&mut self, v_decay: f64) -> &mut Self {
        self.velocity_decay = Some(v_decay);
        self
    }

    fn build(&self) -> Result<Particle<T>, IncompleteParticleBuild> {
        if let Some(object) = self.object.clone() {
            if let Some(position) = self.position.clone() {
                let lower_bound = if let Some(lower) = self.lower_bound.clone() {
                    lower
                } else {
                    vec![-f64::INFINITY; position.len()]
                };
                let upper_bound = if let Some(upper) = self.upper_bound.clone() {
                    upper
                } else {
                    vec![f64::INFINITY; position.len()]
                };
                let velocity = if let Some(vel) = self.velocity.clone() {
                    vel
                } else {
                    // set variance of new particles around data to initial_jitter^2
                    let distr = Normal::new(0.0, 0.25).unwrap();
                    let mut rng = Xoshiro256PlusPlus::from_entropy();
                    let dim = position.len();
                    let mut vel: Vec<f64> = (0..dim)
                        .map(|_| distr.sample(&mut rng))
                        .collect();
                    vel
                };
                let inertia = if let Some(inert) = self.inertia.clone() {
                    inert
                } else {
                    1.0
                };
                let local_weight = if let Some(loc) = self.local_weight.clone() {
                    loc
                } else {
                    1.0
                };
                let global_weight = if let Some(glob) = self.global_weight.clone() {
                    glob
                } else {
                    1.0
                };
                let accept = if let Some(acc) = self.accept_from_logit.clone() {
                    acc
                } else {
                    false
                };
                let vel_decay = if let Some(decay) = self.velocity_decay.clone() {
                    decay
                } else {
                    0.001
                };

                let rng = Xoshiro256PlusPlus::from_entropy();
                let mut particle = Particle {
                    object: object,
                    position: position.clone(),
                    prior_position: position.to_vec(),
                    best_position: position.to_vec(),
                    best_score: f64::INFINITY,
                    score: f64::INFINITY,
                    lower_bound: lower_bound,
                    upper_bound: upper_bound,
                    velocity: velocity.clone(),
                    velocity_decay: vel_decay,
                    inertia: inertia,
                    prior_velocity: velocity.to_vec(),
                    local_weight: local_weight,
                    global_weight: global_weight,
                    accept_from_logit: accept,
                    rng: rng,
                };
                particle.score = particle.evaluate();
                particle.best_score = particle.score;
                Ok(particle)
            } else {
                Err(IncompleteParticleBuild::NoPosition)
            }
        } else {
            Err(IncompleteParticleBuild::NoObjective)
        }
    }
}

pub struct ParticleSwarm<T> {
    particles: Vec<Particle<T>>,
    global_best_position: Vec<f64>,
    global_best_score: f64,
}

impl<T: Objective + Clone + std::marker::Send + std::marker::Sync> ParticleSwarm<T> {
    /// Returns particles whose positions are sampled from
    /// a normal distribution defined by the original start position
    /// plus 1.5 * stepsize.
    pub fn new(
            n_particles: usize,
            data: Vec<f64>,
            lower: Vec<f64>,
            upper: Vec<f64>,
            initial_jitter: f64,
            inertia: f64,
            vel_decay: f64,
            object: T,
            local_weight: f64,
            global_weight: f64,
            accept_from_logit: bool,
    ) -> ParticleSwarm<T>
        where
            T: Objective + Clone,
    {
        // set variance of new particles around data to initial_jitter^2
        let distr = Normal::new(0.0, initial_jitter).unwrap();
        let mut rng = Xoshiro256PlusPlus::from_entropy();

        // instantiate a Vec
        let mut particle_vec: Vec<Particle<T>> = Vec::new();

        // instantiate particles around the actual data
        for i in 0..n_particles {
            //
            let obj_i = object.clone();
            // instantiate the random number generator
            let mut data_vec = data.to_vec();
            if i == 0 {
                // if this is the first particle, place it directly at data_vec
                let particle = ParticleBuilder::new()
                    .set_data(data_vec.to_vec())
                    .set_lower(lower.to_vec())
                    .set_upper(upper.to_vec())
                    .set_inertia(inertia)
                    .set_objective(obj_i)
                    .set_local_weight(local_weight)
                    .set_global_weight(global_weight)
                    .set_accept_from_logit(accept_from_logit)
                    .set_decay(vel_decay)
                    .build().unwrap();
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
                let particle = ParticleBuilder::new()
                    .set_data(data_vec.to_vec())
                    .set_lower(lower.to_vec())
                    .set_upper(upper.to_vec())
                    .set_inertia(inertia)
                    .set_objective(obj_i)
                    .set_local_weight(local_weight)
                    .set_global_weight(global_weight)
                    .set_accept_from_logit(accept_from_logit)
                    .set_decay(vel_decay)
                    .build().unwrap();
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
        ParticleSwarm {
            particles: particle_vec,
            global_best_position: best_pos,
            global_best_score: best_score,
        }
    }

    fn par_iter_mut(&mut self) -> rayon::slice::IterMut<Particle<T>> {
        self.particles.par_iter_mut()
    }

    fn step(&mut self) {
        let global_best = self.global_best_position.to_vec();
        self.par_iter_mut().for_each(|x| {
            x.step(&global_best);
        });
        self.particles.sort_unstable_by_key(|particle| OrderedFloat(particle.score));
        self.global_best_position = self.particles[0].best_position.to_vec();
        self.global_best_score = self.particles[0].best_score;
    }

    /// Returns the number of particles in the ParticleSwarm
    pub fn len(&self) -> usize {self.particles.len()}

    pub fn opt(&mut self, niter: usize)  -> (Vec<f64>, f64) {
        for _ in 0..niter {
            self.step();
        }
        (self.global_best_position.to_vec(), self.global_best_score)
    }
}

impl<T> Optimizer for ParticleSwarm<T>
    where
        T: Objective + Clone + std::marker::Send + std::marker::Sync
{
    fn optimize(&mut self, niter: usize) -> (Vec<f64>, f64) {
        self.opt(niter)
    }
}

struct Particle<T> {
    object: T,
    position: Vec<f64>,
    prior_position: Vec<f64>,
    best_position: Vec<f64>,
    best_score: f64,
    score: f64,
    lower_bound: Vec<f64>,
    upper_bound: Vec<f64>,
    velocity: Vec<f64>,
    prior_velocity: Vec<f64>,
    velocity_decay: f64,
    inertia: f64,
    local_weight: f64,
    global_weight: f64,
    accept_from_logit: bool,
    rng: Xoshiro256PlusPlus,
}

impl<T: Objective + std::marker::Send> Particle<T> {
    pub fn new(
        data: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        inertia: f64,
        local_weight: f64,
        global_weight: f64,
        velocity_decay: f64,
        accept_from_logit: bool,
        object: T,
    ) -> Particle<T>
        where
            T: Objective + std::marker::Send,
    {

        let mut init_rng = Xoshiro256PlusPlus::from_entropy();
        // initialize velocity for each parameter to zero
        let mut v = vec![0.0; data.len()];
        // adjust the starting velocity if we're doing particle swarm
        //let mut rng = thread_rng();
        v.iter_mut()
            .zip(&lower)
            .zip(&upper)
            .for_each(|((vel, low), up)| {
                // draw velocities from uniform dist from +/-(range/40)
                let init_range = (up - low) / 40.0;
                let nudge = init_rng.gen_range(-init_range..init_range);
                *vel = *vel + nudge;
        });
        // copy of velocity to place something into prior_velocity
        let pv = v.to_vec();
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
            velocity: v,
            velocity_decay: velocity_decay,
            inertia: inertia,
            prior_velocity: pv,
            accept_from_logit: accept_from_logit,
            local_weight: local_weight,
            global_weight: global_weight,
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

        // add velocity element-wise to position
        self.position.iter_mut() // mutably iterate over each position
            .zip(&self.velocity) // in lockstep with velocity in each dimension
            .zip(&self.lower_bound) // and the lower bounds for each dim
            .zip(&self.upper_bound) // and the upper bounds for each dim
            .for_each(|(((a, b), c), d)| {
                 // update pos, keeping slightly inward of upper and lower bounds
                *a = (*a + b).clamp(*c+f64::EPSILON, *d-f64::EPSILON)
            });

        // decay velocity by set amount
        let new_vel: Vec<f64> = self.velocity.iter()
            .map(|v| v * (1.0 - self.velocity_decay))
            .collect();
        self.velocity = new_vel;
    }

    /// Set the velocity of the Particle
    fn set_velocity(&mut self, global_best_position: &Vec<f64>) {
        // before we change the velocity,
        // set prior velocity to current velocity
        // this will enable reversion to prior state
        // if we later reject the move
        self.prior_velocity = self.velocity.to_vec();
        // set stochastic element of weights applied to
        // local and global best pos
        // self.rng.gen samples from [0.0, 1.0)
        let r_arr: [f64; 2] = self.rng.gen();
        // set the new velocity
        let new_vel = self.velocity.iter() // iterate over current velocity
            .zip(&self.best_position) // in lockstep with this Particle's
                                      // best position
            .zip(global_best_position) // and the global best position
            .zip(&self.position) // and the current position
            .zip(&self.lower_bound) // and the lower bound
            .zip(&self.upper_bound) // and the upper bound
            // a=vel, b=local_best, c=swarm_best, d=pos,
            // e=lower_bound, f=upper_bound
            .map(|(((((a, b), c), d), e), f)| {
                //let range = f - e;
                let term1 = self.inertia * *a;
                // attraction to the particle's own best gets
                // stronger with distance
                let term2 = self.local_weight * r_arr[0] * (b - d);
                // attraction to the swarms's best gets
                // stronger with distance
                let term3 = self.global_weight * r_arr[1] * (c - d);
                // repulsion from lower bound defined by
                // squared distance to lower bound
                //let term4 = -(range / 100.0) / ((e - d) * (e - d));
                // repulsion from upper bound defined by squared
                // distance to upper bound
                //let term5 = -(range / 100.0) / ((f - d) * (f - d));
                term1 + term2 + term3// + term4 + term5
            }).collect();
        self.velocity = new_vel;
    }

    fn step(&mut self, global_best_position: &Vec<f64>)
        where
            T: Objective + std::marker::Send,
    {
        // set the new velocity
        self.set_velocity(global_best_position);
        // move the particle.
        self.perturb();
        // get the score
        let score = self.evaluate();
        self.update_scores(&score);
    }

    /// Update score fields after a move
    fn update_scores(&mut self, score: &f64) {
        self.score = *score;
        // if this was our best-ever score, update best_score and best_position
        if *score < self.best_score {
            self.best_score = *score;
            self.best_position = self.position.to_vec();
        }
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
            low: &'a Vec<f64>,
            up: &'a Vec<f64>,
            inertia: f64,
            vel_decay: f64,
            init_jitter: f64,
            object: T,
            local_weight: f64,
            global_weight: f64,
            accept_from_logit: bool,
    ) -> ParticleSwarm<T>
        where
            T: Objective + Clone + std::marker::Send + std::marker::Sync,
    {

        let data_vec = vec![0.0, 0.0, 0.0, 0.0, 0.0];

        ParticleSwarm::new(
            n_particles,
            data_vec,
            low.to_vec(),
            up.to_vec(),
            inertia,
            vel_decay,
            init_jitter,
            object,
            local_weight,
            global_weight,
            accept_from_logit,
        )
    }

    fn set_up_particle<T>(
            data_vec: &Vec<f64>,
            low: &Vec<f64>,
            up: &Vec<f64>,
            object: T,
    ) -> Particle<T>
        where
            T: Objective + std::marker::Send + Clone,
    {

        let particle = ParticleBuilder::new()
            .set_data(data_vec.to_vec())
            .set_lower(low.to_vec())
            .set_upper(up.to_vec())
            .set_objective(object)
            .build().unwrap();
        particle
    }

    #[test]
    fn test_eval() {
        let obj = System{};
        let start_data = vec![0.0, 0.0];
        let low = vec![-5.0, -5.0];
        let up = vec![5.0, 5.0];
        let particle = set_up_particle(
            &start_data,
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
    fn test_velocity_only() {
        // Here we test that stepsize 0.0 and velocity [1.0, 1.0] moves particle
        // directionally
        let obj = System{};
        let start_data = vec![0.0, 0.0];
        let step = 0.0;
        let global_best = vec![4.0, 4.0];
        let low = vec![-5.0, -5.0];
        let up = vec![5.0, 5.0];
        let mut particle = set_up_particle(
            &start_data,
            &low,
            &up,
            obj,
        );
        // particle velocity is initialized randomly
        let initial_velocity = particle.velocity.to_vec();
        //assert!(particle.velocity.abs_diff_eq(&array![0.0, 0.0], 1e-6));
        particle.set_velocity(
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

    #[test]
    #[ignore]
    fn test_swarming() {

        let n_particles = 5000;
        let obj = System{};
        let step = 0.25;
        let start = vec![0.0, 0.0];
        let low = vec![-5.0, -5.0];
        let up = vec![5.0, 5.0];
        let inertia = 0.8;
        let local_weight = 0.2;
        let global_weight = 0.8;
        let initial_jitter = &step * 0.25;
        let accept_from_logit = false;
        let vel_decay = 0.0001;

        let niter = 2000;
        let mut swarm = ParticleSwarm::new(
            n_particles,
            start.clone(),
            low,
            up,
            inertia,
            vel_decay,
            initial_jitter,
            obj,
            local_weight,
            global_weight,
            accept_from_logit,
        );
        let opt_params = swarm.optimize(niter);

        let target = vec![1.0,1.0];

        opt_params.0.iter()
            .zip(&start)
            .for_each(|(a,b)| assert_abs_diff_ne!(*a,b));
        opt_params.0.iter()
            .zip(target)
            .for_each(|(a,b)| assert_abs_diff_eq!(*a,b));

        println!("Swarm result: {:?}", opt_params);
    }
}
