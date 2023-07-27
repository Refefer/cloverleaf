//! This provides a simple wrapper around a type to allow for multiple mutable accesses.  
//! The cost of this approach is that two threads can update the same value at the same time which
//! can make for some significant unpleasantness.  Consequently, it should only be used for SGD
//! algorithms (e.g. hogwild optimization) or to access embeddings in parallel where it's know
//! there won't be concurrent accesses.
//!
//! You have been warned.
use std::sync::Arc;
use std::cell::UnsafeCell;
use std::ops::{Deref,DerefMut};


/// Defines the main HogWild structure.  Any item, T, can be mutated across multiple threads.
#[derive(Clone)]
pub struct Hogwild<T>(Arc<UnsafeCell<T>>);

impl<T> Hogwild<T> {
    pub fn new(target: T) -> Hogwild<T> {
        Hogwild(Arc::new(UnsafeCell::new(target)))
    }

    /// Get a mutable reference to T from a shared reference.
    pub fn get(&self) -> &mut T {
        let ptr = self.0.as_ref().get();
        unsafe { &mut *ptr }
    }

    pub fn into_inner(self) -> Option<T> {
        Arc::into_inner(self.0).map(|x| x.into_inner())
    }

}

impl<T> Default for Hogwild<T>
where
    T: Default,
{
    fn default() -> Self {
        Hogwild(Arc::new(UnsafeCell::new(T::default())))
    }
}

impl<T> Deref for Hogwild<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        let ptr = self.0.as_ref().get();
        unsafe { &*ptr }
    }
}

impl<T> DerefMut for Hogwild<T> {
    fn deref_mut(&mut self) -> &mut T {
        let ptr = self.0.as_ref().get();
        unsafe { &mut *ptr }
    }
}

unsafe impl<T> Send for Hogwild<T> {}
unsafe impl<T> Sync for Hogwild<T> {}

