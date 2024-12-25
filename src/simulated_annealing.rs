use approx::{assert_abs_diff_eq,assert_abs_diff_ne};
use rand::prelude::*;
use rand_distr::{Normal,Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use crate::{
    Objective,
    Optimizer,
    IncompleteParticleBuild,
    InvalidParticleBuild,
    logit,
};

pub struct ParticleBuilder<T> {
    object: Option<T>,
    position: Option<Vec<f64>>,
    stepsize: Option<f64>,
    score: Option<f64>,
    lower_bound: Option<Vec<f64>>,
    upper_bound: Option<Vec<f64>>,
    temperature: Option<f64>,
    accept_from_logit: Option<bool>,
    rng: Xoshiro256PlusPlus,
}

impl<T: Objective + Clone> ParticleBuilder<T> {
    pub fn new() -> Self
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
            stepsize: None,
            temperature: None,
            accept_from_logit: None,
            rng: rng,
        }
    }

    pub fn set_data(&mut self, data: Vec<f64>) -> &mut Self {
        self.position = Some(data);
        self
    }

    pub fn set_lower(&mut self, lower: Vec<f64>) -> &mut Self {
        self.lower_bound = Some(lower);
        self
    }

    pub fn set_upper(&mut self, upper: Vec<f64>) -> &mut Self {
        self.upper_bound = Some(upper);
        self
    }

    pub fn set_objective(&mut self, objective: T) ->
        &mut Self
            where
        T: Objective + Clone + std::marker::Send,
    {
        self.object = Some(objective);
        self
    }

    pub fn set_stepsize(&mut self, step: f64) -> &mut Self {
        self.stepsize = Some(step);
        self
    }

    pub fn set_temperature(&mut self, temp: f64) -> &mut Self {
        self.temperature = Some(temp);
        self
    }

    pub fn set_accept_from_logit(&mut self, accept_from_logit: bool) -> &mut Self {
        self.accept_from_logit = Some(accept_from_logit);
        self
    }

    pub fn build(&self) -> Result<Particle<T>, IncompleteParticleBuild> {
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
                let accept = if let Some(acc) = self.accept_from_logit.clone() {
                    acc
                } else {
                    false
                };
                let stepsize = if let Some(step) = self.stepsize.clone() {
                    step
                } else {
                    0.1
                };
                let temperature = if let Some(temp) = self.temperature.clone() {
                    temp
                } else {
                    2.0 
                };

                let rng = Xoshiro256PlusPlus::from_entropy();
                let mut particle = Particle {
                    object: object,
                    position: position.clone(),
                    best_position: position.clone(),
                    prior_position: position.to_vec(),
                    best_score: f64::INFINITY,
                    score: f64::INFINITY,
                    lower_bound: lower_bound,
                    upper_bound: upper_bound,
                    accept_from_logit: accept,
                    temperature: temperature,
                    stepsize: stepsize,
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

#[derive(Debug)]
pub struct Particle<T> {
    pub object: T,
    pub position: Vec<f64>,
    pub prior_position: Vec<f64>,
    pub best_position: Vec<f64>,
    pub best_score: f64,
    pub score: f64,
    pub lower_bound: Vec<f64>,
    pub upper_bound: Vec<f64>,
    pub temperature: f64,
    pub stepsize: f64,
    pub accept_from_logit: bool,
    pub rng: Xoshiro256PlusPlus,
}

impl<T: Objective> Particle<T> {
    pub fn new(
        data: Vec<f64>,
        lower: Vec<f64>,
        upper: Vec<f64>,
        object: T,
        temperature: f64,
        stepsize: f64,
        accept_from_logit: bool,
    ) -> Particle<T>
        where
            T: Objective + Clone + std::marker::Send,
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
            accept_from_logit: accept_from_logit,
        };
        particle.score = particle.evaluate();
        particle.best_score = particle.score;
        particle
    }

    /// Gets the score for this Particle
    pub fn evaluate(
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
    pub fn perturb(&mut self) {

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
        // sample from normal distribution one time
        let jitter_distr = Normal::new(0.0, self.stepsize).unwrap();
        jitter_distr.sample(&mut self.rng)
    }

    pub fn step(&mut self, t_adj: &f64) {
        // move the particle.
        // sets jitter internally
        self.perturb();

        let score = self.evaluate();

        //  Determine whether we accept the move.
        //  if we reject, revert to prior state and perturb again.
        if !self.accept(&score) {
            self.revert();
        } else {
            // Update prior score [and possibly the best score] if we accepted
            self.update_scores(&score);
        }
        // adjust the temperature downward
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
    }
    
    /// Determine whether to accept the new position, or to go back to prior
    /// position and try again. If score is greater that prior score,
    /// return true. If the score is less than prior score, determine whether
    /// to return true probabilistically using the following function:
    ///
    /// `exp(-(score - prior_score)/T)`
    ///
    /// where T is temperature.
    fn accept(&mut self, score: &f64) -> bool {
        if self.accept_from_logit {
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
    /// Defined as
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

pub struct SimulatedAnnealing<T> {
    particles: Vec<Particle<T>>,
    global_best_position: Vec<f64>,
    global_best_score: f64,
    temperature_decay: f64,
}

impl<T: Objective + Clone> SimulatedAnnealing<T> {
    /// Returns particles whose positions are sampled from
    /// a normal distribution defined by the original start position
    /// plus 1.5 * stepsize.
    pub fn new(
            data: Vec<f64>,
            lower: Vec<f64>,
            upper: Vec<f64>,
            temperature: f64,
            temp_decay: f64,
            stepsize: f64,
            object: T,
            accept_from_logit: bool,
    ) -> SimulatedAnnealing<T>
        where
            T: Objective + Clone + std::marker::Send,
    {
        let n_particles = 1;

        // instantiate a Vec
        let mut particle_vec: Vec<Particle<T>> = Vec::new();

        // instantiate particles around the actual data
        for i in 0..n_particles {
            //
            let obj_i = object.clone();
            // instantiate the random number generator
            let mut data_vec = data.to_vec();
            // if this is the first particle, place it directly at data_vec
            let particle = ParticleBuilder::new()
                .set_data(data_vec.to_vec())
                .set_lower(lower.to_vec())
                .set_upper(upper.to_vec())
                .set_objective(obj_i)
                .set_stepsize(stepsize)
                .set_accept_from_logit(accept_from_logit)
                .set_temperature(temperature)
                .build().unwrap();
            particle_vec.push(particle);
        }
        // lowest score is best, so take first one's position and score
        let best_pos = particle_vec[0].best_position.to_vec();
        let best_score = particle_vec[0].best_score;
        SimulatedAnnealing {
            particles: particle_vec,
            global_best_position: best_pos,
            global_best_score: best_score,
            temperature_decay: temp_decay,
        }
    }

    fn step(&mut self) {
        for x in self.particles.iter_mut() {
            x.step(&self.temperature_decay);
        };
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

impl<T> Optimizer for SimulatedAnnealing<T>
    where
        T: Objective + Clone + std::marker::Send,
{
    fn optimize(&mut self, niter: usize) -> (Vec<f64>, f64) {
        self.opt(niter)
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
            temp: f64,
            t_decay: f64,
            step: &'a f64,
            low: &'a Vec<f64>,
            up: &'a Vec<f64>,
            object: T,
    ) -> SimulatedAnnealing<T>
        where
            T: Objective + Clone + std::marker::Send,
    {

        let data_vec = vec![0.0, 0.0, 0.0, 0.0, 0.0];

        SimulatedAnnealing::new(
            data_vec,
            low.to_vec(),
            up.to_vec(),
            temp,
            t_decay,
            *step,
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
            T: Objective + Clone + std::marker::Send,
    {

        let temp = 2.0;
        let particle = ParticleBuilder::new()
            .set_data(data_vec.to_vec())
            .set_lower(low.to_vec())
            .set_upper(up.to_vec())
            .set_objective(object)
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
    fn test_jitter() {
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
    #[ignore]
    fn test_annealing() {
        let obj = System{};
        let step = 0.2;
        let start = vec![0.0, 0.0];
        let low = vec![-5.0, -5.0];
        let up = vec![5.0, 5.0];
        let temp = 1.0;
        let niter = 10000;
        let t_adj = 0.001;

        let mut annealer = SimulatedAnnealing::new(
            start.clone(),
            low,
            up,
            temp,
            t_adj,
            step,
            obj,
            false,
        );

        let opt_params = annealer.optimize(niter);

        let target = vec![1.0,1.0];

        opt_params.0.iter()
            .zip(&start)
            .for_each(|(a,b)| assert_abs_diff_ne!(*a,b));
        opt_params.0.iter()
            .zip(target)
            .for_each(|(a,b)| assert_abs_diff_eq!(*a,b,epsilon=0.03));
        println!("Annealing result: {:?}", opt_params);
    }

}
