use std::sync::Mutex;
use std::time::Duration;

use indicatif::{ProgressBar,ProgressStyle};

pub struct CLProgressBar {
    pb: Option<ProgressBar>,
    message: Mutex<String>
}

impl CLProgressBar {
    pub fn new(work: u64, enabled: bool) -> Self {
        let pb = if enabled {
            let pb = ProgressBar::new(work as u64);
            let style = ProgressStyle::default_bar()
                .template("[{msg}] {wide_bar} ({per_sec}) {pos:>7}/{len:7} - Elapsed: {elapsed_precise}, Remaining: {eta_precise}")
                .expect("Shouldn't fail!");

            pb.set_style(style);

            // Update in separate thread
            pb.enable_steady_tick(Duration::from_millis(200));
            Some(pb)
        } else {
            None
        };

        CLProgressBar {
            pb,
            message: Mutex::new(String::new())
        }
    }

    pub fn update_message<F>(&self, update_message: F) 
    where 
        F: Fn(&mut String) -> () 
    {
        let mut msg = self.message.lock()
            .expect("Mutex poisoned!");

        update_message(&mut *msg);

        if let Some(pb) = &self.pb {
            pb.set_message((*msg).clone());
        }
    }

    pub fn inc(&self, amt: u64) {
        if let Some(pb) = &self.pb {
            pb.inc(amt);
        }
    }

    pub fn finish(&self) {
        if let Some(pb) = &self.pb {
            pb.finish();
        }
    }
}
