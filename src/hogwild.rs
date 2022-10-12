use std::sync::Arc;
use std::cell::UnsafeCell;
use std::ops::{Deref,DerefMut};


#[derive(Clone)]
pub struct Hogwild<T>(Arc<UnsafeCell<T>>);

impl<T> Hogwild<T> {
    pub fn new(target: T) -> Hogwild<T> {
        Hogwild(Arc::new(UnsafeCell::new(target)))
    }

    pub fn get(&self) -> &mut T {
        let ptr = self.0.as_ref().get();
        unsafe { &mut *ptr }
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

