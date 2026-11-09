//! Exponential Moving Average (EMA) filter and rolling coordinate history buffer.

use std::collections::VecDeque;

/// Computes an Exponential Moving Average over a stream of scalar f64 samples.
#[derive(Debug, Clone)]
pub struct EmaFilter {
    alpha: f64,
    current_ema: Option<f64>,
}

impl EmaFilter {
    /// Creates a new `EmaFilter` with `smoothing_alpha` in `(0.0, 1.0]`.
    ///
    /// # Examples
    /// ```
    /// use heater::ema_tracker::EmaFilter;
    /// let filter = EmaFilter::new(0.2).unwrap();
    /// assert_eq!(filter.current_value(), 0.0);
    /// ```
    pub fn new(smoothing_alpha: f64) -> Result<Self, String> {
        if !(0.0 < smoothing_alpha && smoothing_alpha <= 1.0) {
            return Err(format!(
                "Expected smoothing_alpha in (0.0, 1.0], got {smoothing_alpha}"
            ));
        }
        Ok(Self {
            alpha: smoothing_alpha,
            current_ema: None,
        })
    }

    /// Incorporates a new sample and returns the updated EMA estimate.
    ///
    /// # Examples
    /// ```
    /// use heater::ema_tracker::EmaFilter;
    /// let mut filter = EmaFilter::new(0.5).unwrap();
    /// assert_eq!(filter.update_estimate(40.0), 40.0);
    /// assert_eq!(filter.update_estimate(60.0), 50.0);
    /// ```
    pub fn update_estimate(&mut self, sample_value: f64) -> f64 {
        let updated = match self.current_ema {
            None => sample_value,
            Some(prev) => self.alpha * sample_value + (1.0 - self.alpha) * prev,
        };
        self.current_ema = Some(updated);
        updated
    }

    /// Returns the latest EMA value or `0.0` if no samples have been recorded.
    ///
    /// # Examples
    /// ```
    /// use heater::ema_tracker::EmaFilter;
    /// let filter = EmaFilter::new(0.3).unwrap();
    /// assert_eq!(filter.current_value(), 0.0);
    /// ```
    pub fn current_value(&self) -> f64 {
        self.current_ema.unwrap_or(0.0)
    }
}

/// Fixed-capacity ring buffer storing recent telemetry samples and `[x, y]` chart points.
#[derive(Debug, Clone)]
pub struct RollingMetricBuffer {
    max_samples: usize,
    samples: VecDeque<f64>,
}

impl RollingMetricBuffer {
    /// Creates a new `RollingMetricBuffer` with capacity `max_samples >= 2`.
    ///
    /// # Examples
    /// ```
    /// use heater::ema_tracker::RollingMetricBuffer;
    /// let buf = RollingMetricBuffer::new(60).unwrap();
    /// assert_eq!(buf.as_chart_points().len(), 0);
    /// ```
    pub fn new(max_samples: usize) -> Result<Self, String> {
        if max_samples < 2 {
            return Err(format!("Expected max_samples >= 2, got {max_samples}"));
        }
        Ok(Self {
            max_samples,
            samples: VecDeque::with_capacity(max_samples),
        })
    }

    /// Appends a new scalar metric, evicting the oldest entry if full.
    ///
    /// # Examples
    /// ```
    /// use heater::ema_tracker::RollingMetricBuffer;
    /// let mut buf = RollingMetricBuffer::new(3).unwrap();
    /// buf.append_sample(50.0);
    /// assert_eq!(buf.as_chart_points(), vec![(0.0, 50.0)]);
    /// ```
    pub fn append_sample(&mut self, value: f64) {
        if self.samples.len() == self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(value);
    }

    /// Converts buffered samples into `(x, y)` coordinates for `ratatui::widgets::Chart`.
    ///
    /// # Examples
    /// ```
    /// use heater::ema_tracker::RollingMetricBuffer;
    /// let mut buf = RollingMetricBuffer::new(4).unwrap();
    /// buf.append_sample(42.0);
    /// buf.append_sample(45.0);
    /// assert_eq!(buf.as_chart_points(), vec![(0.0, 42.0), (1.0, 45.0)]);
    /// ```
    pub fn as_chart_points(&self) -> Vec<(f64, f64)> {
        self.samples
            .iter()
            .enumerate()
            .map(|(idx, &val)| (idx as f64, val))
            .collect()
    }

    /// Returns `(min, max, latest)` summary statistics for the buffered samples.
    ///
    /// # Examples
    /// ```
    /// use heater::ema_tracker::RollingMetricBuffer;
    /// let mut buf = RollingMetricBuffer::new(4).unwrap();
    /// buf.append_sample(40.0);
    /// buf.append_sample(55.0);
    /// assert_eq!(buf.summary_stats(), (40.0, 55.0, 55.0));
    /// ```
    pub fn summary_stats(&self) -> (f64, f64, f64) {
        if self.samples.is_empty() {
            return (0.0, 0.0, 0.0);
        }
        let min_v = self.samples.iter().copied().fold(f64::INFINITY, f64::min);
        let max_v = self
            .samples
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let last_v = self.samples.back().copied().unwrap_or(0.0);
        (min_v, max_v, last_v)
    }

    /// Returns the maximum capacity of the rolling buffer.
    ///
    /// # Examples
    /// ```
    /// use heater::ema_tracker::RollingMetricBuffer;
    /// let buf = RollingMetricBuffer::new(120).unwrap();
    /// assert_eq!(buf.capacity(), 120);
    /// ```
    pub fn capacity(&self) -> usize {
        self.max_samples
    }
}
