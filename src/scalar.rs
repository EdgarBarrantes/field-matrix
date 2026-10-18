//! [`Scalar`]: what the matrix code needs from its entries.

use core::fmt::{Debug, Display};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use ark_ff::Field;

use crate::counting::Counting;

mod private {
    pub trait Sealed {}
}

/// The field operations the matrix code uses. Implemented for every arkworks field
/// (`ark_ff::Field`) and for [`Counting`], which counts the operations; it can't be implemented
/// outside this crate.
///
/// Users don't need to name it: a `Matrix<F>` with `F: ark_ff::Field` just works.
pub trait Scalar:
    private::Sealed
    + Copy
    + Debug
    + Display
    + Eq
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    /// The additive identity (named apart from arkworks' `ZERO`, so the two never clash).
    const ZERO_ELEMENT: Self;
    /// The multiplicative identity (named apart from arkworks' `ONE`).
    const ONE_ELEMENT: Self;
    /// The multiplicative inverse, or `None` for zero.
    fn invert(&self) -> Option<Self>;
}

impl<F: Field> private::Sealed for F {}

impl<F: Field> Scalar for F {
    const ZERO_ELEMENT: Self = <F as ark_ff::AdditiveGroup>::ZERO;
    const ONE_ELEMENT: Self = <F as Field>::ONE;

    fn invert(&self) -> Option<Self> {
        self.inverse()
    }
}

impl<F: Field> private::Sealed for Counting<F> {}
