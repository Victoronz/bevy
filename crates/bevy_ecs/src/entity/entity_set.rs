use alloc::{
    borrow::{Borrow, Cow},
    collections::{btree_map, btree_set, BTreeSet, BinaryHeap, TryReserveError, VecDeque},
    rc::Rc,
    sync::Arc,
    vec,
};
use bevy_utils::{FixedState, NoOpHash, PassHash};
use core::{
    array,
    cmp::Ordering,
    fmt::{self, Debug, Formatter},
    hash::{BuildHasher, BuildHasherDefault, Hash, Hasher},
    iter::{self, FusedIterator, Map},
    ops::{Deref, Index, RangeBounds},
    option,
    ptr::{self},
    result,
    slice::{self, SliceIndex},
};
use std::{
    array::TryFromSliceError,
    borrow::BorrowMut,
    collections::LinkedList,
    mem::MaybeUninit,
    ops::{
        Bound, DerefMut, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo,
        RangeToInclusive,
    },
};

use super::{Entity, EntityHashSet};

// A trait for entity borrows.
//
// This trait can be thought of as `Borrow<Entity>`, however yielding `Entity` directly.
pub trait EntityBorrow {
    fn entity(&self) -> Entity;
}


// A trait for trustworthy entity borrows.
//
// This trait must only be implemented when Eq, Ord and Hash are equivalent for the borrow and its underlying entity:
// x.entity() == y.entity() should give the same result as x == y.
// In addition, Clone must be reflexive in respect to the aforementioned traits:
// x == x.clone() should be equivalent to x.entity() == x.entity().clone()
// Eq, Ord, Hash and Clone impls are not necessary, but must abide by the above properties if they exist.
pub unsafe trait TrustedEntityBorrow: EntityBorrow {}

impl EntityBorrow for Entity {
    fn entity(&self) -> Entity {
        *self
    }
}

unsafe impl TrustedEntityBorrow for Entity {}

impl<T: EntityBorrow> EntityBorrow for &T {
    fn entity(&self) -> Entity {
        (**self).entity()
    }
}

unsafe impl<T: TrustedEntityBorrow> TrustedEntityBorrow for &T {}

impl<T: EntityBorrow> EntityBorrow for &mut T {
    fn entity(&self) -> Entity {
        (**self).entity()
    }
}

unsafe impl<T: TrustedEntityBorrow> TrustedEntityBorrow for &mut T {}

impl EntityBorrow for Box<Entity> {
    fn entity(&self) -> Entity {
        **self
    }
}

unsafe impl TrustedEntityBorrow for Box<Entity> {}

impl EntityBorrow for Rc<Entity> {
    fn entity(&self) -> Entity {
        **self
    }
}

unsafe impl TrustedEntityBorrow for Rc<Entity> {}

impl EntityBorrow for Arc<Entity> {
    fn entity(&self) -> Entity {
        **self
    }
}

unsafe impl TrustedEntityBorrow for Arc<Entity> {}

pub unsafe trait EntitySet: IntoIterator<Item: TrustedEntityBorrow> {
    // fn difference<E: EntitySet, I: EntitySet>(self, other: E) -> I where Self: Sized;

    // fn symmetric_difference<E: EntitySet, I: EntitySet>(self, other: E) -> I where Self: Sized;

    // fn intersection<E: EntitySet, I: EntitySet>(self, other: E) -> I where Self: Sized;

    // fn union<E: EntitySet, I: EntitySet>(self, other: E) -> I where Self: Sized {
    //     self.into_iter
    // };

    fn is_disjoint<'a, E: EntitySet>(&'a self, other: &'a E) -> bool
    where
        &'a Self: IntoIterator<Item: TrustedEntityBorrow>,
        &'a E: IntoIterator<Item: TrustedEntityBorrow>,
    {
        let other = other
            .into_iter()
            .map(|e| e.entity())
            .collect::<EntityHashSet>();
        self.into_iter().all(|e| !other.contains(&e.entity()))
    }

    fn is_subset<'a, E: EntitySet>(&'a self, other: &'a E) -> bool
    where
        &'a Self: IntoIterator<Item: TrustedEntityBorrow>,
        &'a E: IntoIterator<Item: TrustedEntityBorrow>,
    {
        let other = other
            .into_iter()
            .map(|e| e.entity())
            .collect::<EntityHashSet>();
        self.into_iter().all(|e| other.contains(&e.entity()))
    }

    fn is_superset<'a, E: EntitySet>(&'a self, other: &'a E) -> bool
    where
        &'a Self: IntoIterator<Item: TrustedEntityBorrow>,
        &'a E: IntoIterator<Item: TrustedEntityBorrow>,
    {
        let set = self
            .into_iter()
            .map(|e| e.entity())
            .collect::<EntityHashSet>();
        other.into_iter().all(|e| set.contains(&e.entity()))
    }

    fn is_equal<'a, E: EntitySet>(&'a self, other: &'a E) -> bool
    where
        &'a Self: IntoIterator<Item: TrustedEntityBorrow>,
        &'a E: IntoIterator<Item: TrustedEntityBorrow>,
    {
        let mut other = other
            .into_iter()
            .map(|e| e.entity())
            .collect::<EntityHashSet>();
        self.into_iter().all(|e| other.remove(&e.entity())) && other.is_empty()
    }

    fn own_all(self) -> UniqueEntityIter<Map<Self::IntoIter, fn(Self::Item) -> Entity>>
    where
        Self: Sized,
    {
        UniqueEntityIter {
            iter: self.into_iter().map(|e| e.entity()),
        }
    }

    fn collect_unique_entity_vec(self) -> UniqueEntityVec<Self::Item>
    where
        Self: Sized,
    {
        UniqueEntityVec(self.into_iter().collect())
    }
}

pub unsafe trait ToEntitySet: IntoIterator<Item: TrustedEntityBorrow> {
    fn try_into_unique<I: IntoIterator<Item: TrustedEntityBorrow>>(
        value: I,
    ) -> Result<UniqueEntityIter<vec::IntoIter<I::Item>>, vec::IntoIter<I::Item>> {
        let mut used = EntityHashSet::default();
        let items: Vec<_> = value.into_iter().collect();

        if items.iter().all(move |e| used.insert(e.entity())) {
            return Ok(UniqueEntityIter {
                iter: items.into_iter(),
            });
        }

        Err(items.into_iter())
    }

    // Returns an `EntitySet` that filters out all items that have appeared at least once.
    fn to_unique<I: IntoIterator<Item: TrustedEntityBorrow>>(
        value: I,
    ) -> DuplicateEntityFilterIter<I::IntoIter> {
        DuplicateEntityFilterIter::new(value.into_iter())
    }
}

unsafe impl<T: IntoIterator<Item: TrustedEntityBorrow>> ToEntitySet for T {}




unsafe impl<K: TrustedEntityBorrow, V> EntitySet for btree_map::Keys<'_, K, V> {}

unsafe impl<K: TrustedEntityBorrow, V> EntitySet for btree_map::IntoKeys<K, V> {}

unsafe impl<T: TrustedEntityBorrow> EntitySet for &BTreeSet<T> {}

unsafe impl<T: TrustedEntityBorrow> EntitySet for BTreeSet<T> {}

unsafe impl<T: TrustedEntityBorrow> EntitySet for btree_set::Range<'_, T> {}

unsafe impl<T: TrustedEntityBorrow + Ord> EntitySet for btree_set::Intersection<'_, T> {}

unsafe impl<T: TrustedEntityBorrow + Ord> EntitySet for btree_set::Union<'_, T> {}

unsafe impl<T: TrustedEntityBorrow + Ord> EntitySet for btree_set::Difference<'_, T> {}

unsafe impl<T: TrustedEntityBorrow + Ord> EntitySet for btree_set::SymmetricDifference<'_, T> {}

unsafe impl<T: TrustedEntityBorrow> EntitySet for btree_set::Iter<'_, T> {}

unsafe impl<T: TrustedEntityBorrow> EntitySet for btree_set::IntoIter<T> {}

unsafe impl<T: TrustedEntityBorrow> EntitySet for option::Iter<'_, T> {}

unsafe impl<'a, T: TrustedEntityBorrow> EntitySet for option::IterMut<'_, T> {}

unsafe impl<T: TrustedEntityBorrow> EntitySet for option::IntoIter<T> {}

unsafe impl<T: TrustedEntityBorrow> EntitySet for result::Iter<'_, T> {}

unsafe impl<T: TrustedEntityBorrow> EntitySet for result::IterMut<'_, T> {}

unsafe impl<T: TrustedEntityBorrow> EntitySet for result::IntoIter<T> {}

unsafe impl<T: TrustedEntityBorrow> EntitySet for array::IntoIter<T, 1> {}

unsafe impl<T: TrustedEntityBorrow> EntitySet for array::IntoIter<T, 0> {}

unsafe impl<T: TrustedEntityBorrow, F: FnOnce() -> T> EntitySet for iter::OnceWith<F> {}

unsafe impl<T: TrustedEntityBorrow> EntitySet for iter::Once<T> {}

unsafe impl<T: TrustedEntityBorrow> EntitySet for iter::Empty<T> {}

unsafe impl<I: Iterator<Item: TrustedEntityBorrow> + EntitySet + ?Sized> EntitySet for &mut I {}

unsafe impl<I: Iterator<Item: TrustedEntityBorrow> + EntitySet + ?Sized> EntitySet for Box<I> {}

unsafe impl<'a, T: 'a + TrustedEntityBorrow + Copy, I: Iterator<Item = &'a T> + EntitySet> EntitySet
    for iter::Copied<I>
{
}

unsafe impl<'a, T: 'a + TrustedEntityBorrow + Clone, I: Iterator<Item = &'a T> + EntitySet>
    EntitySet for iter::Cloned<I>
{
}

unsafe impl<
        I: Iterator<Item: TrustedEntityBorrow> + EntitySet,
        P: FnMut(&<I as Iterator>::Item) -> bool,
    > EntitySet for iter::Filter<I, P>
{
}

// Tuple
// impl<I: EntitySet> EntitySet for iter::Enumerate<I> {}

unsafe impl<I: Iterator<Item: TrustedEntityBorrow> + EntitySet> EntitySet for iter::Fuse<I> {}

unsafe impl<I: Iterator<Item: TrustedEntityBorrow> + EntitySet, F: FnMut(&<I as Iterator>::Item)>
    EntitySet for iter::Inspect<I, F>
{
}

unsafe impl<I: DoubleEndedIterator<Item: TrustedEntityBorrow> + EntitySet> EntitySet
    for iter::Rev<I>
{
}

unsafe impl<I: Iterator<Item: TrustedEntityBorrow> + EntitySet> EntitySet for iter::Skip<I> {}

unsafe impl<
        I: Iterator<Item: TrustedEntityBorrow> + EntitySet,
        P: FnMut(&<I as Iterator>::Item) -> bool,
    > EntitySet for iter::SkipWhile<I, P>
{
}

unsafe impl<I: Iterator<Item: TrustedEntityBorrow> + EntitySet> EntitySet for iter::Take<I> {}

unsafe impl<
        I: Iterator<Item: TrustedEntityBorrow> + EntitySet,
        P: FnMut(&<I as Iterator>::Item) -> bool,
    > EntitySet for iter::TakeWhile<I, P>
{
}

unsafe impl<I: Iterator<Item: TrustedEntityBorrow> + EntitySet> EntitySet for iter::StepBy<I> {}

#[derive(Clone, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct UniqueEntityVec<T: TrustedEntityBorrow>(Vec<T>);

impl<T: TrustedEntityBorrow> UniqueEntityVec<T> {
    pub const fn new() -> Self {
        Self(Vec::new())
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    pub unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize) -> Self {
        Self(Vec::from_raw_parts(ptr, length, capacity))
    }

    pub unsafe fn from_vec_unchecked(vec: Vec<T>) -> Self {
        Self(vec)
    }

    pub fn into_inner(self) -> Vec<T> {
        self.0
    }

    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional)
    }

    pub fn reserve_exact(&mut self, additional: usize) {
        self.0.reserve_exact(additional)
    }

    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.0.try_reserve(additional)
    }

    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.0.try_reserve_exact(additional)
    }

    pub fn shrink_to_fit(&mut self) {
        self.0.shrink_to_fit()
    }

    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.0.shrink_to(min_capacity)
    }

    pub fn into_boxed_slice(self) -> Box<UniqueEntitySlice<T>> {
        // SAFETY: UniqueEntitySlice is a transparent wrapper around [T].
        unsafe { UniqueEntitySlice::from_boxed_slice_unchecked(self.0.into_boxed_slice()) }
    }

    pub fn as_slice(&self) -> &UniqueEntitySlice<T> {
        self
    }

    pub fn as_mut_slice(&mut self) -> &mut UniqueEntitySlice<T> {
        self
    }

    pub fn truncate(&mut self, len: usize) {
        self.0.truncate(len)
    }

    pub fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }

    pub unsafe fn set_len(&mut self, new_len: usize) {
        self.0.set_len(new_len)
    }

    pub fn swap_remove(&mut self, index: usize) -> T {
        self.0.swap_remove(index)
    }

    pub unsafe fn insert(&mut self, index: usize, element: T) {
        self.0.insert(index, element)
    }

    pub fn retain<F>(&mut self, f: F)
    where
        F: FnMut(&T) -> bool,
    {
        self.0.retain(f)
    }

    pub unsafe fn retain_mut<F>(&mut self, f: F)
    where
        F: FnMut(&mut T) -> bool,
    {
        self.0.retain_mut(f)
    }

    pub fn dedup_by_key<F, K>(&mut self, key: F)
    where
        F: FnMut(&mut T) -> K,
        K: PartialEq,
    {
        self.0.dedup_by_key(key)
    }

    pub fn dedup_by<F>(&mut self, same_bucket: F)
    where
        F: FnMut(&mut T, &mut T) -> bool,
    {
        self.0.dedup_by(same_bucket)
    }

    pub unsafe fn push(&mut self, value: T) {
        self.0.push(value)
    }

    pub unsafe fn append(&mut self, other: &mut UniqueEntityVec<T>) {
        self.0.append(&mut other.0)
    }

    pub fn pop(&mut self) -> Option<T> {
        self.0.pop()
    }

    pub fn drain<R>(&mut self, range: R) -> vec::Drain<'_, T>
    where
        R: RangeBounds<usize>,
    {
        self.0.drain(range)
    }

    pub fn clear(&mut self) {
        self.0.clear()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn split_off(&mut self, at: usize) -> Self {
        Self(self.0.split_off(at))
    }

    pub unsafe fn resize_with<F>(&mut self, new_len: usize, f: F)
    where
        F: FnMut() -> T,
    {
        self.0.resize_with(new_len, f)
    }

    pub fn leak<'a>(self) -> &'a mut UniqueEntitySlice<T> {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.0.leak()) }
    }

    pub unsafe fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        self.0.spare_capacity_mut()
    }

    pub unsafe fn splice<R, I>(
        &mut self,
        range: R,
        replace_with: I,
    ) -> vec::Splice<'_, <I as IntoIterator>::IntoIter>
    where
        R: RangeBounds<usize>,
        I: IntoIterator<Item = T>,
    {
        self.0.splice(range, replace_with)
    }
}

impl<T: TrustedEntityBorrow + Clone> UniqueEntityVec<T> {
    pub unsafe fn extend_from_slice(&mut self, other: &UniqueEntitySlice<T>) {
        self.0.extend_from_slice(other)
    }
}

impl<T: TrustedEntityBorrow> Default for UniqueEntityVec<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<T: TrustedEntityBorrow> Deref for UniqueEntityVec<T> {
    type Target = UniqueEntitySlice<T>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(&self.0) }
    }
}

impl<T: TrustedEntityBorrow> DerefMut for UniqueEntityVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(&mut self.0) }
    }
}

impl<'a, T: TrustedEntityBorrow> IntoIterator for &'a UniqueEntityVec<T>
where
    &'a T: TrustedEntityBorrow,
{
    type Item = &'a T;

    type IntoIter = UniqueEntityIter<slice::Iter<'a, T>>;

    fn into_iter(self) -> Self::IntoIter {
        UniqueEntityIter {
            iter: self.0.iter(),
        }
    }
}

impl<T: TrustedEntityBorrow> IntoIterator for UniqueEntityVec<T> {
    type Item = T;

    type IntoIter = UniqueEntityIter<vec::IntoIter<T>>;

    fn into_iter(self) -> Self::IntoIter {
        UniqueEntityIter {
            iter: self.0.into_iter(),
        }
    }
}

unsafe impl<T: TrustedEntityBorrow> EntitySet for &UniqueEntityVec<T> {}

unsafe impl<T: TrustedEntityBorrow> EntitySet for UniqueEntityVec<T> {}

impl<T: TrustedEntityBorrow> AsMut<UniqueEntitySlice<T>> for UniqueEntityVec<T> {
    fn as_mut(&mut self) -> &mut UniqueEntitySlice<T> {
        self
    }
}

impl<T: TrustedEntityBorrow> AsMut<UniqueEntityVec<T>> for UniqueEntityVec<T> {
    fn as_mut(&mut self) -> &mut UniqueEntityVec<T> {
        self
    }
}

impl<T: TrustedEntityBorrow> AsRef<UniqueEntitySlice<T>> for UniqueEntityVec<T> {
    fn as_ref(&self) -> &UniqueEntitySlice<T> {
        self
    }
}

impl<T: TrustedEntityBorrow> AsRef<UniqueEntityVec<T>> for UniqueEntityVec<T> {
    fn as_ref(&self) -> &UniqueEntityVec<T> {
        self
    }
}

impl<T: TrustedEntityBorrow> Borrow<UniqueEntitySlice<T>> for UniqueEntityVec<T> {
    fn borrow(&self) -> &UniqueEntitySlice<T> {
        self
    }
}

impl<T: TrustedEntityBorrow> BorrowMut<UniqueEntitySlice<T>> for UniqueEntityVec<T> {
    fn borrow_mut(&mut self) -> &mut UniqueEntitySlice<T> {
        self
    }
}

impl<T: TrustedEntityBorrow + Clone> From<&[T; 1]> for UniqueEntityVec<T> {
    fn from(value: &[T; 1]) -> Self {
        Self(Vec::from(value))
    }
}

impl<T: TrustedEntityBorrow + Clone> From<&[T; 0]> for UniqueEntityVec<T> {
    fn from(value: &[T; 0]) -> Self {
        Self(Vec::from(value))
    }
}

impl<T: TrustedEntityBorrow + Clone> From<&mut [T; 1]> for UniqueEntityVec<T> {
    fn from(value: &mut [T; 1]) -> Self {
        Self(Vec::from(value))
    }
}

impl<T: TrustedEntityBorrow + Clone> From<&mut [T; 0]> for UniqueEntityVec<T> {
    fn from(value: &mut [T; 0]) -> Self {
        Self(Vec::from(value))
    }
}

impl<T: TrustedEntityBorrow> From<[T; 1]> for UniqueEntityVec<T> {
    fn from(value: [T; 1]) -> Self {
        Self(Vec::from(value))
    }
}

impl<T: TrustedEntityBorrow> From<[T; 0]> for UniqueEntityVec<T> {
    fn from(value: [T; 0]) -> Self {
        Self(Vec::from(value))
    }
}

impl<T: TrustedEntityBorrow + Clone, const N: usize> From<&UniqueEntityArray<T, N>>
    for UniqueEntityVec<T>
{
    fn from(value: &UniqueEntityArray<T, N>) -> Self {
        Self(Vec::from(value.0.clone()))
    }
}

impl<T: TrustedEntityBorrow + Clone, const N: usize> From<&mut UniqueEntityArray<T, N>>
    for UniqueEntityVec<T>
{
    fn from(value: &mut UniqueEntityArray<T, N>) -> Self {
        Self(Vec::from(value.0.clone()))
    }
}

impl<T: TrustedEntityBorrow, const N: usize> From<UniqueEntityArray<T, N>> for UniqueEntityVec<T> {
    fn from(value: UniqueEntityArray<T, N>) -> Self {
        Self(Vec::from(value.0))
    }
}

impl<T: TrustedEntityBorrow> From<Box<UniqueEntitySlice<T>>> for UniqueEntityVec<T> {
    fn from(value: Box<UniqueEntitySlice<T>>) -> Self {
        Self(value.into_vec())
    }
}

impl<T: TrustedEntityBorrow> From<Cow<'_, UniqueEntitySlice<T>>> for UniqueEntityVec<T>
where
    UniqueEntitySlice<T>: ToOwned<Owned = UniqueEntityVec<T>>,
{
    fn from(value: Cow<UniqueEntitySlice<T>>) -> Self {
        value.into_owned()
    }
}

impl<'a, T: TrustedEntityBorrow + Clone> From<UniqueEntityVec<T>> for Cow<'a, [T]> {
    fn from(value: UniqueEntityVec<T>) -> Self {
        Cow::from(value.0)
    }
}

impl<T: TrustedEntityBorrow> From<UniqueEntityVec<T>> for Arc<[T]> {
    fn from(value: UniqueEntityVec<T>) -> Self {
        Arc::from(value.0)
    }
}

impl<T: TrustedEntityBorrow + Ord> From<UniqueEntityVec<T>> for BinaryHeap<T> {
    fn from(value: UniqueEntityVec<T>) -> Self {
        BinaryHeap::from(value.0)
    }
}

impl<T: TrustedEntityBorrow> From<UniqueEntityVec<T>> for Box<[T]> {
    fn from(value: UniqueEntityVec<T>) -> Self {
        Box::from(value.0)
    }
}

impl<T: TrustedEntityBorrow> From<UniqueEntityVec<T>> for Rc<[T]> {
    fn from(value: UniqueEntityVec<T>) -> Self {
        Rc::from(value.0)
    }
}

impl<T: TrustedEntityBorrow> From<UniqueEntityVec<T>> for VecDeque<T> {
    fn from(value: UniqueEntityVec<T>) -> Self {
        VecDeque::from(value.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U> PartialEq<&[U]> for UniqueEntityVec<T> {
    fn eq(&self, other: &&[U]) -> bool {
        self.0.eq(other)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U: TrustedEntityBorrow> PartialEq<&UniqueEntitySlice<U>>
    for UniqueEntityVec<T>
{
    fn eq(&self, other: &&UniqueEntitySlice<U>) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U> PartialEq<&mut [U]> for UniqueEntityVec<T> {
    fn eq(&self, other: &&mut [U]) -> bool {
        self.0.eq(other)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U: TrustedEntityBorrow>
    PartialEq<&mut UniqueEntitySlice<U>> for UniqueEntityVec<T>
{
    fn eq(&self, other: &&mut UniqueEntitySlice<U>) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U, const N: usize> PartialEq<&[U; N]>
    for UniqueEntityVec<T>
{
    fn eq(&self, other: &&[U; N]) -> bool {
        self.0.eq(other)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U: TrustedEntityBorrow, const N: usize>
    PartialEq<&UniqueEntityArray<U, N>> for UniqueEntityVec<T>
{
    fn eq(&self, other: &&UniqueEntityArray<U, N>) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U, const N: usize> PartialEq<&mut [U; N]>
    for UniqueEntityVec<T>
{
    fn eq(&self, other: &&mut [U; N]) -> bool {
        self.0.eq(&**other)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U: TrustedEntityBorrow, const N: usize>
    PartialEq<&mut UniqueEntityArray<U, N>> for UniqueEntityVec<T>
{
    fn eq(&self, other: &&mut UniqueEntityArray<U, N>) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U> PartialEq<[U]> for UniqueEntityVec<T> {
    fn eq(&self, other: &[U]) -> bool {
        self.0.eq(other)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U: TrustedEntityBorrow> PartialEq<UniqueEntitySlice<U>>
    for UniqueEntityVec<T>
{
    fn eq(&self, other: &UniqueEntitySlice<U>) -> bool {
        self.0.eq(&**other)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U, const N: usize> PartialEq<[U; N]>
    for UniqueEntityVec<T>
{
    fn eq(&self, other: &[U; N]) -> bool {
        self.0.eq(other)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U: TrustedEntityBorrow, const N: usize>
    PartialEq<UniqueEntityArray<U, N>> for UniqueEntityVec<T>
{
    fn eq(&self, other: &UniqueEntityArray<U, N>) -> bool {
        self.0.eq(&***other)
    }
}

impl<T: PartialEq<U>, U: TrustedEntityBorrow> PartialEq<UniqueEntityVec<U>> for &[T] {
    fn eq(&self, other: &UniqueEntityVec<U>) -> bool {
        self.eq(&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U: TrustedEntityBorrow> PartialEq<UniqueEntityVec<U>>
    for &UniqueEntitySlice<T>
{
    fn eq(&self, other: &UniqueEntityVec<U>) -> bool {
        (&self.0).eq(&other.0)
    }
}

impl<T: PartialEq<U>, U: TrustedEntityBorrow> PartialEq<UniqueEntityVec<U>> for &mut [T] {
    fn eq(&self, other: &UniqueEntityVec<U>) -> bool {
        self.eq(&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U: TrustedEntityBorrow> PartialEq<UniqueEntityVec<U>>
    for &mut UniqueEntitySlice<T>
{
    fn eq(&self, other: &UniqueEntityVec<U>) -> bool {
        (&self.0).eq(&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U: TrustedEntityBorrow> PartialEq<UniqueEntityVec<U>>
    for [T]
{
    fn eq(&self, other: &UniqueEntityVec<U>) -> bool {
        self.eq(&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U: TrustedEntityBorrow> PartialEq<UniqueEntityVec<U>>
    for UniqueEntitySlice<T>
{
    fn eq(&self, other: &UniqueEntityVec<U>) -> bool {
        (&self.0).eq(&other.0)
    }
}

impl<T: PartialEq<U> + Clone, U: TrustedEntityBorrow> PartialEq<UniqueEntityVec<U>>
    for Cow<'_, [T]>
{
    fn eq(&self, other: &UniqueEntityVec<U>) -> bool {
        self.eq(&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U> + Clone, U: TrustedEntityBorrow>
    PartialEq<UniqueEntityVec<U>> for Cow<'_, UniqueEntitySlice<T>>
{
    fn eq(&self, other: &UniqueEntityVec<U>) -> bool {
        (&self.0).eq(&other.0)
    }
}

impl<T: PartialEq<U>, U: TrustedEntityBorrow> PartialEq<UniqueEntityVec<U>> for VecDeque<T> {
    fn eq(&self, other: &UniqueEntityVec<U>) -> bool {
        self.eq(&other.0)
    }
}
impl<T: PartialEq<U>, U: TrustedEntityBorrow> PartialEq<UniqueEntityVec<U>> for Vec<T> {
    fn eq(&self, other: &UniqueEntityVec<U>) -> bool {
        self.eq(&other.0)
    }
}

impl<T: TrustedEntityBorrow, const N: usize> TryFrom<UniqueEntityVec<T>> for Box<[T; N]> {
    type Error = UniqueEntityVec<T>;

    fn try_from(value: UniqueEntityVec<T>) -> Result<Self, Self::Error> {
        Box::try_from(value.0).map_err(UniqueEntityVec)
    }
}

impl<T: TrustedEntityBorrow, const N: usize> TryFrom<UniqueEntityVec<T>>
    for Box<UniqueEntityArray<T, N>>
{
    type Error = UniqueEntityVec<T>;

    fn try_from(value: UniqueEntityVec<T>) -> Result<Self, Self::Error> {
        Box::try_from(value.0)
            .map(|v|
                // SAFETY: All elements in the original Vec are unique.
                unsafe { UniqueEntityArray::from_boxed_array_unchecked(v) })
            .map_err(UniqueEntityVec)
    }
}

impl<T: TrustedEntityBorrow, const N: usize> TryFrom<UniqueEntityVec<T>> for [T; N] {
    type Error = UniqueEntityVec<T>;

    fn try_from(value: UniqueEntityVec<T>) -> Result<Self, Self::Error> {
        <[T; N] as TryFrom<Vec<T>>>::try_from(value.0).map_err(UniqueEntityVec)
    }
}

impl<T: TrustedEntityBorrow, const N: usize> TryFrom<UniqueEntityVec<T>>
    for UniqueEntityArray<T, N>
{
    type Error = UniqueEntityVec<T>;

    fn try_from(value: UniqueEntityVec<T>) -> Result<Self, Self::Error> {
        <[T; N] as TryFrom<Vec<T>>>::try_from(value.0)
            .map(|v|
            // SAFETY: All elements in the original Vec are unique.
            unsafe { UniqueEntityArray::from_array_unchecked(v) })
            .map_err(UniqueEntityVec)
    }
}

impl<T: TrustedEntityBorrow> From<BTreeSet<T>> for UniqueEntityVec<T> {
    fn from(value: BTreeSet<T>) -> Self {
        Self(value.into_iter().collect::<Vec<T>>())
    }
}

impl<T: TrustedEntityBorrow> FromIterator<T> for UniqueEntityVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut unique_vec = Self::new();
        for item in iter {
            if unique_vec.iter().any(|e| e.entity().eq(&item.entity())) {
                continue;
            }
            unique_vec.0.push(item)
        }
        unique_vec
    }
}

impl<T: TrustedEntityBorrow> Extend<T> for UniqueEntityVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let mut unique_vec = Self::new();
        for item in iter {
            if unique_vec.iter().any(|e| e.entity().eq(&item.entity())) {
                continue;
            }
            unique_vec.0.push(item)
        }
    }
}

impl<'a, T: TrustedEntityBorrow + Copy + 'a> Extend<&'a T> for UniqueEntityVec<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        let mut unique_vec = Self::new();
        for item in iter {
            if unique_vec.iter().any(|e| e.entity().eq(&item.entity())) {
                continue;
            }
            unique_vec.0.push(*item)
        }
    }
}

impl<T: TrustedEntityBorrow> Index<(Bound<usize>, Bound<usize>)> for UniqueEntityVec<T> {
    type Output = UniqueEntitySlice<T>;
    fn index(&self, key: (Bound<usize>, Bound<usize>)) -> &Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow> Index<Range<usize>> for UniqueEntityVec<T> {
    type Output = UniqueEntitySlice<T>;
    fn index(&self, key: Range<usize>) -> &Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow> Index<RangeFrom<usize>> for UniqueEntityVec<T> {
    type Output = UniqueEntitySlice<T>;
    fn index(&self, key: RangeFrom<usize>) -> &Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow> Index<RangeFull> for UniqueEntityVec<T> {
    type Output = UniqueEntitySlice<T>;
    fn index(&self, key: RangeFull) -> &Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow> Index<RangeInclusive<usize>> for UniqueEntityVec<T> {
    type Output = UniqueEntitySlice<T>;
    fn index(&self, key: RangeInclusive<usize>) -> &Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow> Index<RangeTo<usize>> for UniqueEntityVec<T> {
    type Output = UniqueEntitySlice<T>;
    fn index(&self, key: RangeTo<usize>) -> &Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow> Index<RangeToInclusive<usize>> for UniqueEntityVec<T> {
    type Output = UniqueEntitySlice<T>;
    fn index(&self, key: RangeToInclusive<usize>) -> &Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow> Index<usize> for UniqueEntityVec<T> {
    type Output = T;
    fn index(&self, key: usize) -> &T {
        self.0.index(key)
    }
}

impl<T: TrustedEntityBorrow> IndexMut<(Bound<usize>, Bound<usize>)> for UniqueEntityVec<T> {
    fn index_mut(&mut self, key: (Bound<usize>, Bound<usize>)) -> &mut Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow> IndexMut<Range<usize>> for UniqueEntityVec<T> {
    fn index_mut(&mut self, key: Range<usize>) -> &mut Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow> IndexMut<RangeFrom<usize>> for UniqueEntityVec<T> {
    fn index_mut(&mut self, key: RangeFrom<usize>) -> &mut Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow> IndexMut<RangeFull> for UniqueEntityVec<T> {
    fn index_mut(&mut self, key: RangeFull) -> &mut Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow> IndexMut<RangeInclusive<usize>> for UniqueEntityVec<T> {
    fn index_mut(&mut self, key: RangeInclusive<usize>) -> &mut Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow> IndexMut<RangeTo<usize>> for UniqueEntityVec<T> {
    fn index_mut(&mut self, key: RangeTo<usize>) -> &mut Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow> IndexMut<RangeToInclusive<usize>> for UniqueEntityVec<T> {
    fn index_mut(&mut self, key: RangeToInclusive<usize>) -> &mut Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

#[repr(transparent)]
#[derive(Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct UniqueEntitySlice<T: TrustedEntityBorrow>([T]);

impl<'a, T: TrustedEntityBorrow> UniqueEntitySlice<T> {
    pub const unsafe fn from_slice_unchecked(slice: &[T]) -> &Self {
        // SAFETY: UniqueEntitySlice is a transparent wrapper around [T].
        unsafe { &*(ptr::from_ref(slice) as *const Self) }
    }

    pub const unsafe fn from_slice_unchecked_mut(slice: &mut [T]) -> &mut Self {
        // SAFETY: UniqueEntitySlice is a transparent wrapper around [T].
        unsafe { &mut *(ptr::from_mut(slice) as *mut Self) }
    }

    pub unsafe fn from_boxed_slice_unchecked(slice: Box<[T]>) -> Box<Self> {
        // SAFETY: UniqueEntitySlice is a transparent wrapper around [T].
        unsafe { Box::from_raw(Box::into_raw(slice) as *mut Self) }
    }

    pub fn into_boxed_inner(self: Box<Self>) -> Box<[T]> {
        // SAFETY: UniqueEntitySlice is a transparent wrapper around [T].
        unsafe { Box::from_raw(Box::into_raw(self) as *mut [T]) }
    }

    pub unsafe fn from_arc_slice_unchecked(slice: Arc<[T]>) -> Arc<Self> {
        // SAFETY: UniqueEntitySlice is a transparent wrapper around [T].
        unsafe { Arc::from_raw(Arc::into_raw(slice) as *mut Self) }
    }

    pub fn into_arc_inner(self: Arc<Self>) -> Arc<[T]> {
        // SAFETY: UniqueEntitySlice is a transparent wrapper around [T].
        unsafe { Arc::from_raw(Arc::into_raw(self) as *mut [T]) }
    }

    pub unsafe fn from_rc_slice_unchecked(slice: Rc<[T]>) -> Rc<Self> {
        // SAFETY: UniqueEntitySlice is a transparent wrapper around [T].
        unsafe { Rc::from_raw(Rc::into_raw(slice) as *mut Self) }
    }

    pub fn into_rc_inner(self: Rc<Self>) -> Rc<[T]> {
        // SAFETY: UniqueEntitySlice is a transparent wrapper around [T].
        unsafe { Rc::from_raw(Rc::into_raw(self) as *mut [T]) }
    }

    pub const fn split_first(&self) -> Option<(&T, &UniqueEntitySlice<T>)> {
        let Some((first, rest)) = self.0.split_first() else {
            return None;
        };
        // SAFETY: All elements in the original slice are unique.
        Some((first, unsafe { Self::from_slice_unchecked(rest) }))
    }

    pub const fn split_last(&self) -> Option<(&T, &UniqueEntitySlice<T>)> {
        let Some((last, rest)) = self.0.split_last() else {
            return None;
        };
        // SAFETY: All elements in the original slice are unique.
        Some((last, unsafe { Self::from_slice_unchecked(rest) }))
    }

    pub const fn first_chunk<const N: usize>(&self) -> Option<&UniqueEntityArray<T, N>> {
        let Some(chunk) = self.0.first_chunk() else {
            return None;
        };
        // SAFETY: All elements in the original slice are unique.
        Some(unsafe { UniqueEntityArray::from_array_ref_unchecked(chunk) })
    }

    pub const fn split_first_chunk<const N: usize>(
        &self,
    ) -> Option<(&UniqueEntityArray<T, N>, &UniqueEntitySlice<T>)> {
        let Some((chunk, rest)) = self.0.split_first_chunk() else {
            return None;
        };
        // SAFETY: All elements in the original slice are unique.
        unsafe {
            Some((
                UniqueEntityArray::from_array_ref_unchecked(chunk),
                Self::from_slice_unchecked(rest),
            ))
        }
    }

    pub const fn split_last_chunk<const N: usize>(
        &self,
    ) -> Option<(&UniqueEntitySlice<T>, &UniqueEntityArray<T, N>)> {
        let Some((rest, chunk)) = self.0.split_last_chunk() else {
            return None;
        };
        // SAFETY: All elements in the original slice are unique.
        unsafe {
            Some((
                Self::from_slice_unchecked(rest),
                UniqueEntityArray::from_array_ref_unchecked(chunk),
            ))
        }
    }

    pub const fn last_chunk<const N: usize>(&self) -> Option<&UniqueEntityArray<T, N>> {
        let Some(chunk) = self.0.last_chunk() else {
            return None;
        };
        // SAFETY: All elements in the original slice are unique.
        Some(unsafe { UniqueEntityArray::from_array_ref_unchecked(chunk) })
    }

    pub fn get<I>(&self, index: I) -> Option<&Self>
    where
        Self: Index<I>,
        I: SliceIndex<[T], Output = [T]>,
    {
        self.0.get(index).map(|slice|
            // SAFETY: All elements in the original slice are unique.
            unsafe { Self::from_slice_unchecked(slice) })
    }

    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut Self>
    where
        Self: Index<I>,
        I: SliceIndex<[T], Output = [T]>,
    {
        self.0.get_mut(index).map(|slice|
            // SAFETY: All elements in the original slice are unique.
            unsafe { Self::from_slice_unchecked_mut(slice) })
    }

    pub unsafe fn get_unchecked<I>(&self, index: I) -> &Self
    where
        Self: Index<I>,
        I: SliceIndex<[T], Output = [T]>,
    {
        // SAFETY: All elements in the original slice are unique.
        unsafe { Self::from_slice_unchecked(self.0.get_unchecked(index)) }
    }

    pub unsafe fn get_unchecked_mut<I>(&mut self, index: I) -> &mut Self
    where
        Self: Index<I>,
        I: SliceIndex<[T], Output = [T]>,
    {
        // SAFETY: All elements in the original slice are unique.
        unsafe { Self::from_slice_unchecked_mut(self.0.get_unchecked_mut(index)) }
    }

    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }

    pub const fn as_mut_ptr_range(&mut self) -> Range<*mut T> {
        self.0.as_mut_ptr_range()
    }

    pub fn swap(&mut self, a: usize, b: usize) {
        self.0.swap(a, b)
    }

    pub fn reverse(&mut self) {
        self.0.reverse()
    }

    pub fn iter(&self) -> UniqueEntityIter<slice::Iter<'_, T>> {
        UniqueEntityIter {
            iter: self.0.iter(),
        }
    }

    pub fn windows(&self, size: usize) -> Windows<'_, T> {
        UniqueEntitySliceIter {
            iter: self.0.windows(size),
        }
    }

    pub fn chunks(&self, chunk_size: usize) -> Chunks<'_, T> {
        UniqueEntitySliceIter {
            iter: self.0.chunks(chunk_size),
        }
    }

    pub fn chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<'_, T> {
        UniqueEntitySliceIterMut {
            iter: self.0.chunks_mut(chunk_size),
        }
    }

    pub fn chunks_exact(&self, chunk_size: usize) -> ChunksExact<'_, T> {
        UniqueEntitySliceIter {
            iter: self.0.chunks_exact(chunk_size),
        }
    }

    pub fn chunks_exact_mut(&mut self, chunk_size: usize) -> ChunksExactMut<'_, T> {
        UniqueEntitySliceIterMut {
            iter: self.0.chunks_exact_mut(chunk_size),
        }
    }

    pub fn rchunks(&self, chunk_size: usize) -> RChunks<'_, T> {
        UniqueEntitySliceIter {
            iter: self.0.rchunks(chunk_size),
        }
    }

    pub fn rchunks_mut(&mut self, chunk_size: usize) -> RChunksMut<'_, T> {
        UniqueEntitySliceIterMut {
            iter: self.0.rchunks_mut(chunk_size),
        }
    }

    pub fn rchunks_exact(&self, chunk_size: usize) -> RChunksExact<'_, T> {
        UniqueEntitySliceIter {
            iter: self.0.rchunks_exact(chunk_size),
        }
    }

    pub fn rchunks_exact_mut(&mut self, chunk_size: usize) -> RChunksExactMut<'_, T> {
        UniqueEntitySliceIterMut {
            iter: self.0.rchunks_exact_mut(chunk_size),
        }
    }

    pub fn chunk_by<F>(&self, pred: F) -> ChunkBy<'_, T, F>
    where
        F: FnMut(&T, &T) -> bool,
    {
        UniqueEntitySliceIter {
            iter: self.0.chunk_by(pred),
        }
    }

    pub fn chunk_by_mut<F>(&mut self, pred: F) -> ChunkByMut<'_, T, F>
    where
        F: FnMut(&T, &T) -> bool,
    {
        UniqueEntitySliceIterMut {
            iter: self.0.chunk_by_mut(pred),
        }
    }

    pub const fn split_at(&self, mid: usize) -> (&UniqueEntitySlice<T>, &UniqueEntitySlice<T>) {
        let (left, right) = self.0.split_at(mid);
        // SAFETY: All elements in the original slice are unique.
        unsafe {
            (
                Self::from_slice_unchecked(left),
                Self::from_slice_unchecked(right),
            )
        }
    }

    pub const fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> (&mut UniqueEntitySlice<T>, &mut UniqueEntitySlice<T>) {
        let (left, right) = self.0.split_at_mut(mid);
        // SAFETY: All elements in the original slice are unique.
        unsafe {
            (
                Self::from_slice_unchecked_mut(left),
                Self::from_slice_unchecked_mut(right),
            )
        }
    }

    pub const unsafe fn split_at_unchecked(
        &self,
        mid: usize,
    ) -> (&UniqueEntitySlice<T>, &UniqueEntitySlice<T>) {
        let (left, right) = self.0.split_at_unchecked(mid);
        // SAFETY: All elements in the original slice are unique.
        unsafe {
            (
                Self::from_slice_unchecked(left),
                Self::from_slice_unchecked(right),
            )
        }
    }

    pub const unsafe fn split_at_mut_unchecked(
        &mut self,
        mid: usize,
    ) -> (&mut UniqueEntitySlice<T>, &mut UniqueEntitySlice<T>) {
        let (left, right) = self.0.split_at_mut_unchecked(mid);
        // SAFETY: All elements in the original slice are unique.
        unsafe {
            (
                Self::from_slice_unchecked_mut(left),
                Self::from_slice_unchecked_mut(right),
            )
        }
    }

    pub const fn split_at_checked(
        &self,
        mid: usize,
    ) -> Option<(&UniqueEntitySlice<T>, &UniqueEntitySlice<T>)> {
        let Some((left, right)) = self.0.split_at_checked(mid) else {
            return None;
        };
        // SAFETY: All elements in the original slice are unique.
        unsafe {
            Some((
                Self::from_slice_unchecked(left),
                Self::from_slice_unchecked(right),
            ))
        }
    }

    pub const fn split_at_mut_checked(
        &mut self,
        mid: usize,
    ) -> Option<(&mut UniqueEntitySlice<T>, &mut UniqueEntitySlice<T>)> {
        let Some((left, right)) = self.0.split_at_mut_checked(mid) else {
            return None;
        };
        // SAFETY: All elements in the original slice are unique.
        unsafe {
            Some((
                Self::from_slice_unchecked_mut(left),
                Self::from_slice_unchecked_mut(right),
            ))
        }
    }

    pub fn split<F>(&self, pred: F) -> Split<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        UniqueEntitySliceIter {
            iter: self.0.split(pred),
        }
    }

    pub fn split_mut<F>(&mut self, pred: F) -> SplitMut<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        UniqueEntitySliceIterMut {
            iter: self.0.split_mut(pred),
        }
    }

    pub fn split_inclusive<F>(&self, pred: F) -> SplitInclusive<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        UniqueEntitySliceIter {
            iter: self.0.split_inclusive(pred),
        }
    }

    pub fn split_inclusive_mut<F>(&mut self, pred: F) -> SplitInclusiveMut<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        UniqueEntitySliceIterMut {
            iter: self.0.split_inclusive_mut(pred),
        }
    }

    pub fn rsplit<F>(&self, pred: F) -> RSplit<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        UniqueEntitySliceIter {
            iter: self.0.rsplit(pred),
        }
    }

    pub fn rsplit_mut<F>(&mut self, pred: F) -> RSplitMut<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        UniqueEntitySliceIterMut {
            iter: self.0.rsplit_mut(pred),
        }
    }

    pub fn splitn<F>(&self, n: usize, pred: F) -> SplitN<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        UniqueEntitySliceIter {
            iter: self.0.splitn(n, pred),
        }
    }

    pub fn splitn_mut<F>(&mut self, n: usize, pred: F) -> SplitNMut<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        UniqueEntitySliceIterMut {
            iter: self.0.splitn_mut(n, pred),
        }
    }

    pub fn rsplitn<F>(&self, n: usize, pred: F) -> RSplitN<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        UniqueEntitySliceIter {
            iter: self.0.rsplitn(n, pred),
        }
    }

    pub fn rsplitn_mut<F>(&mut self, n: usize, pred: F) -> RSplitNMut<'_, T, F>
    where
        F: FnMut(&T) -> bool,
    {
        UniqueEntitySliceIterMut {
            iter: self.0.rsplitn_mut(n, pred),
        }
    }

    pub fn sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.0.sort_unstable();
    }

    pub fn sort_unstable_by<F>(&mut self, compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        self.0.sort_unstable_by(compare);
    }

    pub fn sort_unstable_by_key<K, F>(&mut self, f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.0.sort_unstable_by_key(f);
    }

    pub fn rotate_left(&mut self, mid: usize) {
        self.0.rotate_left(mid);
    }

    pub fn rotate_right(&mut self, mid: usize) {
        self.0.rotate_right(mid);
    }

    pub unsafe fn fill_with<F>(&mut self, f: F)
    where
        F: FnMut() -> T,
    {
        self.0.fill_with(f)
    }

    pub fn copy_from_slice(&mut self, src: &UniqueEntitySlice<T>)
    where
        T: Copy,
    {
        self.0.copy_from_slice(src);
    }

    pub unsafe fn swap_with_slice(&mut self, other: &mut UniqueEntitySlice<T>) {
        (&mut self.0).swap_with_slice(&mut other.0);
    }

    pub fn sort(&mut self)
    where
        T: Ord,
    {
        self.0.sort();
    }

    pub fn sort_by<F>(&mut self, compare: F)
    where
        F: FnMut(&T, &T) -> Ordering,
    {
        self.0.sort_by(compare);
    }

    pub fn sort_by_key<K, F>(&mut self, f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.0.sort_by_key(f);
    }

    pub fn sort_by_cached_key<K, F>(&mut self, f: F)
    where
        F: FnMut(&T) -> K,
        K: Ord,
    {
        self.0.sort_by_cached_key(f);
    }

    pub fn into_vec(self: Box<UniqueEntitySlice<T>>) -> Vec<T> {
        unsafe {
            let len = self.len();
            Vec::from_raw_parts(Box::into_raw(self) as *mut T, len, len)
        }
    }
}

impl<'a, 'b, T: TrustedEntityBorrow + 'a> UniqueEntitySlice<T> {
    pub unsafe fn cast_slice_of_unique_entity_slices(
        slice: &'b [&'a [T]],
    ) -> &'b [&'a UniqueEntitySlice<T>] {
        // SAFETY: All elements in the original iterator are unique slices.
        unsafe { &*(ptr::from_ref(slice) as *const [&UniqueEntitySlice<T>]) }
    }

    pub unsafe fn cast_slice_of_unique_entity_slices_mut(
        slice: &'b mut [&'a mut [T]],
    ) -> &'b mut [&'a mut UniqueEntitySlice<T>] {
        // SAFETY: All elements in the original iterator are unique slices.
        unsafe { &mut *(ptr::from_mut(slice) as *mut [&mut UniqueEntitySlice<T>]) }
    }
}

impl<'a, T: TrustedEntityBorrow> IntoIterator for &'a UniqueEntitySlice<T> {
    type Item = &'a T;

    type IntoIter = UniqueEntityIter<slice::Iter<'a, T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T: TrustedEntityBorrow> IntoIterator for &'a Box<UniqueEntitySlice<T>> {
    type Item = &'a T;

    type IntoIter = UniqueEntityIter<slice::Iter<'a, T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<T: TrustedEntityBorrow> IntoIterator for Box<UniqueEntitySlice<T>> {
    type Item = T;

    type IntoIter = UniqueEntityIter<vec::IntoIter<T>>;

    fn into_iter(self) -> Self::IntoIter {
        UniqueEntityIter {
            iter: self.into_vec().into_iter(),
        }
    }
}

impl<T: TrustedEntityBorrow> Deref for UniqueEntitySlice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: TrustedEntityBorrow> AsRef<[T]> for UniqueEntitySlice<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T: TrustedEntityBorrow> AsRef<Self> for UniqueEntitySlice<T> {
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<T: TrustedEntityBorrow> AsMut<Self> for UniqueEntitySlice<T> {
    fn as_mut(&mut self) -> &mut Self {
        self
    }
}

impl<T: TrustedEntityBorrow> Borrow<[T]> for UniqueEntitySlice<T> {
    fn borrow(&self) -> &[T] {
        self
    }
}

impl<T: TrustedEntityBorrow + Clone> Clone for Box<UniqueEntitySlice<T>> {
    fn clone(&self) -> Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_boxed_slice_unchecked(self.to_vec().into_boxed_slice()) }
    }
}

impl<T: TrustedEntityBorrow> Default for &UniqueEntitySlice<T> {
    fn default() -> Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(Default::default()) }
    }
}

impl<T: TrustedEntityBorrow> Default for &mut UniqueEntitySlice<T> {
    fn default() -> Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(Default::default()) }
    }
}

impl<T: TrustedEntityBorrow> Default for Box<UniqueEntitySlice<T>> {
    fn default() -> Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_boxed_slice_unchecked(Default::default()) }
    }
}

impl<T: TrustedEntityBorrow + Clone> From<&UniqueEntitySlice<T>> for Box<UniqueEntitySlice<T>> {
    fn from(value: &UniqueEntitySlice<T>) -> Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_boxed_slice_unchecked(value.0.into()) }
    }
}

impl<T: TrustedEntityBorrow + Clone> From<&UniqueEntitySlice<T>> for Arc<UniqueEntitySlice<T>> {
    fn from(value: &UniqueEntitySlice<T>) -> Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_arc_slice_unchecked(value.0.into()) }
    }
}

impl<T: TrustedEntityBorrow + Clone> From<&UniqueEntitySlice<T>> for Rc<UniqueEntitySlice<T>> {
    fn from(value: &UniqueEntitySlice<T>) -> Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_rc_slice_unchecked(value.0.into()) }
    }
}

impl<'a, T: TrustedEntityBorrow + Clone> From<&'a UniqueEntitySlice<T>>
    for Cow<'a, UniqueEntitySlice<T>>
{
    fn from(value: &'a UniqueEntitySlice<T>) -> Self {
        Cow::Borrowed(value)
    }
}

impl<T: TrustedEntityBorrow + Clone, const N: usize> From<UniqueEntityArray<T, N>>
    for Box<UniqueEntitySlice<T>>
{
    fn from(value: UniqueEntityArray<T, N>) -> Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_boxed_slice_unchecked(Box::new(value.0)) }
    }
}

impl<'a, T: TrustedEntityBorrow + Clone> From<Cow<'a, UniqueEntitySlice<T>>>
    for Box<UniqueEntitySlice<T>>
{
    fn from(value: Cow<'a, UniqueEntitySlice<T>>) -> Self {
        match value {
            Cow::Borrowed(slice) => Box::from(slice),
            Cow::Owned(slice) => Box::from(slice),
        }
    }
}

impl<T: TrustedEntityBorrow> From<UniqueEntityVec<T>> for Box<UniqueEntitySlice<T>> {
    fn from(value: UniqueEntityVec<T>) -> Self {
        value.into_boxed_slice()
    }
}

impl<T: TrustedEntityBorrow> FromIterator<T> for Box<UniqueEntitySlice<T>> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        iter.into_iter()
            .collect::<UniqueEntityVec<T>>()
            .into_boxed_slice()
    }
}

impl<T: PartialEq<U>, U: TrustedEntityBorrow, const N: usize> PartialEq<&UniqueEntitySlice<U>>
    for [T; N]
{
    fn eq(&self, other: &&UniqueEntitySlice<U>) -> bool {
        self.eq(&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U: TrustedEntityBorrow, const N: usize>
    PartialEq<&UniqueEntitySlice<U>> for UniqueEntityArray<T, N>
{
    fn eq(&self, other: &&UniqueEntitySlice<U>) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T: PartialEq<U> + Clone, U: TrustedEntityBorrow> PartialEq<&UniqueEntitySlice<U>>
    for Cow<'_, [T]>
{
    fn eq(&self, other: &&UniqueEntitySlice<U>) -> bool {
        self.eq(&&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U> + Clone, U: TrustedEntityBorrow>
    PartialEq<&UniqueEntitySlice<U>> for Cow<'_, UniqueEntitySlice<T>>
{
    fn eq(&self, other: &&UniqueEntitySlice<U>) -> bool {
        (&self.0).eq(&other.0)
    }
}

impl<T: PartialEq<U>, U: TrustedEntityBorrow> PartialEq<&UniqueEntitySlice<U>> for Vec<T> {
    fn eq(&self, other: &&UniqueEntitySlice<U>) -> bool {
        self.eq(&other.0)
    }
}

impl<T: PartialEq<U>, U: TrustedEntityBorrow> PartialEq<&UniqueEntitySlice<U>> for VecDeque<T> {
    fn eq(&self, other: &&UniqueEntitySlice<U>) -> bool {
        self.eq(&&other.0)
    }
}

impl<T: PartialEq<U>, U: TrustedEntityBorrow, const N: usize> PartialEq<&mut UniqueEntitySlice<U>>
    for [T; N]
{
    fn eq(&self, other: &&mut UniqueEntitySlice<U>) -> bool {
        self.eq(&other.0)
    }
}

impl<T: PartialEq<U> + Clone, U: TrustedEntityBorrow> PartialEq<&mut UniqueEntitySlice<U>>
    for Cow<'_, [T]>
{
    fn eq(&self, other: &&mut UniqueEntitySlice<U>) -> bool {
        self.eq(&&**other)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U> + Clone, U: TrustedEntityBorrow>
    PartialEq<&mut UniqueEntitySlice<U>> for Cow<'_, UniqueEntitySlice<T>>
{
    fn eq(&self, other: &&mut UniqueEntitySlice<U>) -> bool {
        (&self.0).eq(&other.0)
    }
}

impl<T: PartialEq<U>, U: TrustedEntityBorrow> PartialEq<&mut UniqueEntitySlice<U>> for Vec<T> {
    fn eq(&self, other: &&mut UniqueEntitySlice<U>) -> bool {
        self.eq(&other.0)
    }
}

impl<T: PartialEq<U>, U: TrustedEntityBorrow> PartialEq<&mut UniqueEntitySlice<U>> for VecDeque<T> {
    fn eq(&self, other: &&mut UniqueEntitySlice<U>) -> bool {
        self.eq(&&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U: TrustedEntityBorrow> PartialEq<UniqueEntitySlice<U>>
    for [T]
{
    fn eq(&self, other: &UniqueEntitySlice<U>) -> bool {
        self.eq(&other.0)
    }
}

impl<T: PartialEq<U>, U: TrustedEntityBorrow, const N: usize> PartialEq<UniqueEntitySlice<U>>
    for [T; N]
{
    fn eq(&self, other: &UniqueEntitySlice<U>) -> bool {
        self.eq(&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U: TrustedEntityBorrow, const N: usize>
    PartialEq<UniqueEntitySlice<U>> for UniqueEntityArray<T, N>
{
    fn eq(&self, other: &UniqueEntitySlice<U>) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U: TrustedEntityBorrow> PartialEq<UniqueEntitySlice<U>>
    for Vec<T>
{
    fn eq(&self, other: &UniqueEntitySlice<U>) -> bool {
        self.eq(&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U, const N: usize> PartialEq<[U; N]>
    for &UniqueEntitySlice<T>
{
    fn eq(&self, other: &[U; N]) -> bool {
        self.0.eq(other)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U, const N: usize> PartialEq<[U; N]>
    for &mut UniqueEntitySlice<T>
{
    fn eq(&self, other: &[U; N]) -> bool {
        self.0.eq(other)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U, const N: usize> PartialEq<[U; N]>
    for UniqueEntitySlice<T>
{
    fn eq(&self, other: &[U; N]) -> bool {
        self.0.eq(other)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U: TrustedEntityBorrow, const N: usize>
    PartialEq<UniqueEntityArray<U, N>> for &UniqueEntitySlice<T>
{
    fn eq(&self, other: &UniqueEntityArray<U, N>) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U: TrustedEntityBorrow, const N: usize>
    PartialEq<UniqueEntityArray<U, N>> for &mut UniqueEntitySlice<T>
{
    fn eq(&self, other: &UniqueEntityArray<U, N>) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U: TrustedEntityBorrow, const N: usize>
    PartialEq<UniqueEntityArray<U, N>> for UniqueEntitySlice<T>
{
    fn eq(&self, other: &UniqueEntityArray<U, N>) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U> PartialEq<Vec<U>> for &UniqueEntitySlice<T> {
    fn eq(&self, other: &Vec<U>) -> bool {
        (&self.0).eq(other)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U> PartialEq<Vec<U>> for &mut UniqueEntitySlice<T> {
    fn eq(&self, other: &Vec<U>) -> bool {
        (&self.0).eq(other)
    }
}

impl<T: TrustedEntityBorrow + PartialEq<U>, U> PartialEq<Vec<U>> for UniqueEntitySlice<T> {
    fn eq(&self, other: &Vec<U>) -> bool {
        (&self.0).eq(other)
    }
}

impl<T: TrustedEntityBorrow + Clone> ToOwned for UniqueEntitySlice<T> {
    type Owned = UniqueEntityVec<T>;

    fn to_owned(&self) -> Self::Owned {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntityVec::from_vec_unchecked(self.0.to_owned()) }
    }
}

impl<'a, T: TrustedEntityBorrow + Copy, const N: usize> TryFrom<&'a UniqueEntitySlice<T>>
    for &'a UniqueEntityArray<T, N>
{
    type Error = TryFromSliceError;

    fn try_from(value: &'a UniqueEntitySlice<T>) -> Result<Self, Self::Error> {
        <&[T; N]>::try_from(&value.0)
            .map(|array| unsafe { UniqueEntityArray::from_array_ref_unchecked(array) })
    }
}

impl<T: TrustedEntityBorrow + Copy, const N: usize> TryFrom<&UniqueEntitySlice<T>>
    for UniqueEntityArray<T, N>
{
    type Error = TryFromSliceError;

    fn try_from(value: &UniqueEntitySlice<T>) -> Result<Self, Self::Error> {
        <&Self>::try_from(value).copied()
    }
}

impl<T: TrustedEntityBorrow + Copy, const N: usize> TryFrom<&mut UniqueEntitySlice<T>>
    for UniqueEntityArray<T, N>
{
    type Error = TryFromSliceError;

    fn try_from(value: &mut UniqueEntitySlice<T>) -> Result<Self, Self::Error> {
        <Self>::try_from(&*value)
    }
}

impl<T: TrustedEntityBorrow> Index<(Bound<usize>, Bound<usize>)> for UniqueEntitySlice<T> {
    type Output = Self;
    fn index(&self, key: (Bound<usize>, Bound<usize>)) -> &Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { Self::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow> Index<Range<usize>> for UniqueEntitySlice<T> {
    type Output = Self;
    fn index(&self, key: Range<usize>) -> &Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { Self::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow> Index<RangeFrom<usize>> for UniqueEntitySlice<T> {
    type Output = Self;
    fn index(&self, key: RangeFrom<usize>) -> &Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { Self::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow> Index<RangeFull> for UniqueEntitySlice<T> {
    type Output = Self;
    fn index(&self, key: RangeFull) -> &Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { Self::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow> Index<RangeInclusive<usize>> for UniqueEntitySlice<T> {
    type Output = UniqueEntitySlice<T>;
    fn index(&self, key: RangeInclusive<usize>) -> &Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { Self::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow> Index<RangeTo<usize>> for UniqueEntitySlice<T> {
    type Output = UniqueEntitySlice<T>;
    fn index(&self, key: RangeTo<usize>) -> &Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { Self::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow> Index<RangeToInclusive<usize>> for UniqueEntitySlice<T> {
    type Output = UniqueEntitySlice<T>;
    fn index(&self, key: RangeToInclusive<usize>) -> &Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { Self::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow> Index<usize> for UniqueEntitySlice<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self.0[index]
    }
}

impl<T: TrustedEntityBorrow> IndexMut<(Bound<usize>, Bound<usize>)> for UniqueEntitySlice<T> {
    fn index_mut(&mut self, key: (Bound<usize>, Bound<usize>)) -> &mut Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { Self::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow> IndexMut<Range<usize>> for UniqueEntitySlice<T> {
    fn index_mut(&mut self, key: Range<usize>) -> &mut Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { Self::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow> IndexMut<RangeFrom<usize>> for UniqueEntitySlice<T> {
    fn index_mut(&mut self, key: RangeFrom<usize>) -> &mut Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { Self::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow> IndexMut<RangeFull> for UniqueEntitySlice<T> {
    fn index_mut(&mut self, key: RangeFull) -> &mut Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { Self::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow> IndexMut<RangeInclusive<usize>> for UniqueEntitySlice<T> {
    fn index_mut(&mut self, key: RangeInclusive<usize>) -> &mut Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { Self::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow> IndexMut<RangeTo<usize>> for UniqueEntitySlice<T> {
    fn index_mut(&mut self, key: RangeTo<usize>) -> &mut Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { Self::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow> IndexMut<RangeToInclusive<usize>> for UniqueEntitySlice<T> {
    fn index_mut(&mut self, key: RangeToInclusive<usize>) -> &mut Self {
        // SAFETY: All elements in the original slice are unique.
        unsafe { Self::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct UniqueEntityArray<T: TrustedEntityBorrow, const N: usize>([T; N]);

impl<T: TrustedEntityBorrow, const N: usize> UniqueEntityArray<T, N> {
    pub const unsafe fn from_array_unchecked(array: [T; N]) -> Self {
        Self(array)
    }

    pub const unsafe fn from_array_ref_unchecked(array: &[T; N]) -> &Self {
        // SAFETY: UniqueEntityArray is a transparent wrapper around [T; N].
        unsafe { &*(ptr::from_ref(array).cast()) }
    }

    pub const unsafe fn from_array_ref_unchecked_mut(array: &mut [T; N]) -> &mut Self {
        // SAFETY: UniqueEntityArray is a transparent wrapper around [T; N].
        unsafe { &mut *(ptr::from_mut(array).cast()) }
    }

    pub unsafe fn from_boxed_array_unchecked(array: Box<[T; N]>) -> Box<Self> {
        // SAFETY: UniqueEntityArray is a transparent wrapper around [T; N].
        unsafe { Box::from_raw(Box::into_raw(array).cast()) }
    }

    pub fn into_boxed_inner(self: Box<Self>) -> Box<[T; N]> {
        // SAFETY: UniqueEntityArray is a transparent wrapper around [T; N].
        unsafe { Box::from_raw(Box::into_raw(self).cast()) }
    }

    pub fn into_inner(self) -> [T; N] {
        self.0
    }

    pub const fn as_slice(&self) -> &UniqueEntitySlice<T> {
        // SAFETY: All elements in the original array are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.0.as_slice()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut UniqueEntitySlice<T> {
        // SAFETY: All elements in the original array are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.0.as_mut_slice()) }
    }

    pub fn each_ref(&self) -> UniqueEntityArray<&T, N> {
        UniqueEntityArray(self.0.each_ref())
    }
}

impl<T: TrustedEntityBorrow, const N: usize> Deref for UniqueEntityArray<T, N> {
    type Target = UniqueEntitySlice<T>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: All elements in the original array are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(&self.0) }
    }
}

impl<T: TrustedEntityBorrow, const N: usize> DerefMut for UniqueEntityArray<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: All elements in the original array are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(&mut self.0) }
    }
}
impl<T: TrustedEntityBorrow> Default for UniqueEntityArray<T, 0> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<'a, T: TrustedEntityBorrow, const N: usize> IntoIterator for &'a UniqueEntityArray<T, N> {
    type Item = &'a T;

    type IntoIter = UniqueEntityIter<slice::Iter<'a, T>>;

    fn into_iter(self) -> Self::IntoIter {
        UniqueEntityIter {
            iter: self.0.iter(),
        }
    }
}

impl<T: TrustedEntityBorrow, const N: usize> IntoIterator for UniqueEntityArray<T, N> {
    type Item = T;

    type IntoIter = UniqueEntityIter<array::IntoIter<T, N>>;

    fn into_iter(self) -> Self::IntoIter {
        UniqueEntityIter {
            iter: self.0.into_iter(),
        }
    }
}

impl<T: TrustedEntityBorrow, const N: usize> AsRef<UniqueEntitySlice<T>>
    for UniqueEntityArray<T, N>
{
    fn as_ref(&self) -> &UniqueEntitySlice<T> {
        self
    }
}

impl<T: TrustedEntityBorrow, const N: usize> AsMut<UniqueEntitySlice<T>>
    for UniqueEntityArray<T, N>
{
    fn as_mut(&mut self) -> &mut UniqueEntitySlice<T> {
        self
    }
}

impl<T: TrustedEntityBorrow, const N: usize> Borrow<UniqueEntitySlice<T>>
    for UniqueEntityArray<T, N>
{
    fn borrow(&self) -> &UniqueEntitySlice<T> {
        self
    }
}

impl<T: TrustedEntityBorrow, const N: usize> BorrowMut<UniqueEntitySlice<T>>
    for UniqueEntityArray<T, N>
{
    fn borrow_mut(&mut self) -> &mut UniqueEntitySlice<T> {
        self
    }
}

impl<T: TrustedEntityBorrow, const N: usize> Index<(Bound<usize>, Bound<usize>)>
    for UniqueEntityArray<T, N>
{
    type Output = UniqueEntitySlice<T>;
    fn index(&self, key: (Bound<usize>, Bound<usize>)) -> &Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow, const N: usize> Index<Range<usize>> for UniqueEntityArray<T, N> {
    type Output = UniqueEntitySlice<T>;
    fn index(&self, key: Range<usize>) -> &Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow, const N: usize> Index<RangeFrom<usize>> for UniqueEntityArray<T, N> {
    type Output = UniqueEntitySlice<T>;
    fn index(&self, key: RangeFrom<usize>) -> &Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow, const N: usize> Index<RangeFull> for UniqueEntityArray<T, N> {
    type Output = UniqueEntitySlice<T>;
    fn index(&self, key: RangeFull) -> &Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow, const N: usize> Index<RangeInclusive<usize>>
    for UniqueEntityArray<T, N>
{
    type Output = UniqueEntitySlice<T>;
    fn index(&self, key: RangeInclusive<usize>) -> &Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow, const N: usize> Index<RangeTo<usize>> for UniqueEntityArray<T, N> {
    type Output = UniqueEntitySlice<T>;
    fn index(&self, key: RangeTo<usize>) -> &Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow, const N: usize> Index<RangeToInclusive<usize>>
    for UniqueEntityArray<T, N>
{
    type Output = UniqueEntitySlice<T>;
    fn index(&self, key: RangeToInclusive<usize>) -> &Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<T: TrustedEntityBorrow, const N: usize> Index<usize> for UniqueEntityArray<T, N> {
    type Output = T;
    fn index(&self, key: usize) -> &T {
        self.0.index(key)
    }
}

impl<T: TrustedEntityBorrow, const N: usize> IndexMut<(Bound<usize>, Bound<usize>)>
    for UniqueEntityArray<T, N>
{
    fn index_mut(&mut self, key: (Bound<usize>, Bound<usize>)) -> &mut Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow, const N: usize> IndexMut<Range<usize>> for UniqueEntityArray<T, N> {
    fn index_mut(&mut self, key: Range<usize>) -> &mut Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow, const N: usize> IndexMut<RangeFrom<usize>>
    for UniqueEntityArray<T, N>
{
    fn index_mut(&mut self, key: RangeFrom<usize>) -> &mut Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow, const N: usize> IndexMut<RangeFull> for UniqueEntityArray<T, N> {
    fn index_mut(&mut self, key: RangeFull) -> &mut Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow, const N: usize> IndexMut<RangeInclusive<usize>>
    for UniqueEntityArray<T, N>
{
    fn index_mut(&mut self, key: RangeInclusive<usize>) -> &mut Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow, const N: usize> IndexMut<RangeTo<usize>> for UniqueEntityArray<T, N> {
    fn index_mut(&mut self, key: RangeTo<usize>) -> &mut Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow, const N: usize> IndexMut<RangeToInclusive<usize>>
    for UniqueEntityArray<T, N>
{
    fn index_mut(&mut self, key: RangeToInclusive<usize>) -> &mut Self::Output {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<T: TrustedEntityBorrow + Clone> From<&[T; 1]> for UniqueEntityArray<T, 1> {
    fn from(value: &[T; 1]) -> Self {
        Self(value.clone())
    }
}

impl<T: TrustedEntityBorrow + Clone> From<&[T; 0]> for UniqueEntityArray<T, 0> {
    fn from(value: &[T; 0]) -> Self {
        Self(value.clone())
    }
}

impl<T: TrustedEntityBorrow + Clone> From<&mut [T; 1]> for UniqueEntityArray<T, 1> {
    fn from(value: &mut [T; 1]) -> Self {
        Self(value.clone())
    }
}

impl<T: TrustedEntityBorrow + Clone> From<&mut [T; 0]> for UniqueEntityArray<T, 0> {
    fn from(value: &mut [T; 0]) -> Self {
        Self(value.clone())
    }
}

impl<T: TrustedEntityBorrow> From<[T; 1]> for UniqueEntityArray<T, 1> {
    fn from(value: [T; 1]) -> Self {
        Self(value)
    }
}

impl<T: TrustedEntityBorrow> From<[T; 0]> for UniqueEntityArray<T, 0> {
    fn from(value: [T; 0]) -> Self {
        Self(value)
    }
}

// impl From<UniqueEntityArray> for Tuples from size 0-12

impl<T: TrustedEntityBorrow + Ord, const N: usize> From<UniqueEntityArray<T, N>> for BTreeSet<T> {
    fn from(value: UniqueEntityArray<T, N>) -> Self {
        BTreeSet::from(value.0)
    }
}

impl<T: TrustedEntityBorrow + Ord, const N: usize> From<UniqueEntityArray<T, N>> for BinaryHeap<T> {
    fn from(value: UniqueEntityArray<T, N>) -> Self {
        BinaryHeap::from(value.0)
    }
}

impl<T: TrustedEntityBorrow, const N: usize> From<UniqueEntityArray<T, N>> for LinkedList<T> {
    fn from(value: UniqueEntityArray<T, N>) -> Self {
        LinkedList::from(value.0)
    }
}

impl<T: TrustedEntityBorrow, const N: usize> From<UniqueEntityArray<T, N>> for Vec<T> {
    fn from(value: UniqueEntityArray<T, N>) -> Self {
        Vec::from(value.0)
    }
}

impl<T: TrustedEntityBorrow, const N: usize> From<UniqueEntityArray<T, N>> for VecDeque<T> {
    fn from(value: UniqueEntityArray<T, N>) -> Self {
        VecDeque::from(value.0)
    }
}

impl<T: PartialEq<U>, U: TrustedEntityBorrow, const N: usize> PartialEq<&UniqueEntityArray<U, N>>
    for Vec<T>
{
    fn eq(&self, other: &&UniqueEntityArray<U, N>) -> bool {
        self.eq(&other.0)
    }
}
impl<T: PartialEq<U>, U: TrustedEntityBorrow, const N: usize> PartialEq<&UniqueEntityArray<U, N>>
    for VecDeque<T>
{
    fn eq(&self, other: &&UniqueEntityArray<U, N>) -> bool {
        self.eq(&other.0)
    }
}

impl<T: PartialEq<U>, U: TrustedEntityBorrow, const N: usize>
    PartialEq<&mut UniqueEntityArray<U, N>> for VecDeque<T>
{
    fn eq(&self, other: &&mut UniqueEntityArray<U, N>) -> bool {
        self.eq(&other.0)
    }
}

impl<T: PartialEq<U>, U: TrustedEntityBorrow, const N: usize> PartialEq<UniqueEntityArray<U, N>>
    for Vec<T>
{
    fn eq(&self, other: &UniqueEntityArray<U, N>) -> bool {
        self.eq(&other.0)
    }
}
impl<T: PartialEq<U>, U: TrustedEntityBorrow, const N: usize> PartialEq<UniqueEntityArray<U, N>>
    for VecDeque<T>
{
    fn eq(&self, other: &UniqueEntityArray<U, N>) -> bool {
        self.eq(&other.0)
    }
}

/// An iterator that yields unique entities/entity borrows.
pub struct UniqueEntityIter<I: Iterator<Item: TrustedEntityBorrow>> {
    pub(crate) iter: I,
}

impl<I: Iterator<Item: TrustedEntityBorrow>> UniqueEntityIter<I> {
    pub unsafe fn from_iterator_unchecked(iter: I) -> Self {
        Self { iter }
    }
}

impl<'a, T: TrustedEntityBorrow> UniqueEntityIter<slice::Iter<'a, T>> {
    pub fn as_slice(&self) -> &'a UniqueEntitySlice<T> {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.iter.as_slice()) }
    }
}

impl<'a, T: TrustedEntityBorrow> UniqueEntityIter<vec::Drain<'a, T>> {
    pub fn as_slice(&self) -> &UniqueEntitySlice<T> {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.iter.as_slice()) }
    }
}

impl<T: TrustedEntityBorrow> UniqueEntityIter<vec::IntoIter<T>> {
    pub fn as_slice(&self) -> &UniqueEntitySlice<T> {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.iter.as_slice()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut UniqueEntitySlice<T> {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.iter.as_mut_slice()) }
    }
}

impl<T: TrustedEntityBorrow, const N: usize> UniqueEntityIter<array::IntoIter<T, N>> {
    pub fn as_slice(&self) -> &UniqueEntitySlice<T> {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.iter.as_slice()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut UniqueEntitySlice<T> {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.iter.as_mut_slice()) }
    }
}

impl<I: Iterator<Item: TrustedEntityBorrow>> Iterator for UniqueEntityIter<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I: ExactSizeIterator<Item: TrustedEntityBorrow>> ExactSizeIterator for UniqueEntityIter<I> {}

impl<I: DoubleEndedIterator<Item: TrustedEntityBorrow>> DoubleEndedIterator
    for UniqueEntityIter<I>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<I: FusedIterator<Item: TrustedEntityBorrow>> FusedIterator for UniqueEntityIter<I> {}

unsafe impl<I: Iterator<Item: TrustedEntityBorrow>> EntitySet for UniqueEntityIter<I> {}

impl<T, I: Iterator<Item: TrustedEntityBorrow> + AsRef<[T]>> AsRef<[T]> for UniqueEntityIter<I> {
    fn as_ref(&self) -> &[T] {
        self.iter.as_ref()
    }
}

impl<'a, T: TrustedEntityBorrow> AsRef<UniqueEntitySlice<T>>
    for UniqueEntityIter<vec::Drain<'a, T>>
{
    fn as_ref(&self) -> &UniqueEntitySlice<T> {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.iter.as_ref()) }
    }
}

impl<T: TrustedEntityBorrow> AsRef<UniqueEntitySlice<T>> for UniqueEntityIter<vec::IntoIter<T>> {
    fn as_ref(&self) -> &UniqueEntitySlice<T> {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.iter.as_ref()) }
    }
}

impl<'a, T: TrustedEntityBorrow> AsRef<UniqueEntitySlice<T>>
    for UniqueEntityIter<slice::Iter<'a, T>>
{
    fn as_ref(&self) -> &UniqueEntitySlice<T> {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.iter.as_ref()) }
    }
}

impl<'a, T: TrustedEntityBorrow> AsRef<UniqueEntitySlice<T>>
    for UniqueEntityIter<slice::IterMut<'a, T>>
{
    fn as_ref(&self) -> &UniqueEntitySlice<T> {
        // SAFETY: All elements in the original slice are unique.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.iter.as_ref()) }
    }
}

impl<I: Iterator<Item: TrustedEntityBorrow> + Default> Default for UniqueEntityIter<I> {
    fn default() -> Self {
        Self {
            iter: Default::default(),
        }
    }
}

impl<I: Iterator<Item: TrustedEntityBorrow> + Clone> Clone for UniqueEntityIter<I> {
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
        }
    }
}

impl<I: Iterator<Item: TrustedEntityBorrow>> Debug for UniqueEntityIter<I> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("UniqueEntityIter").finish()
    }
}

pub struct DuplicateEntityFilterIter<I: Iterator<Item: TrustedEntityBorrow>> {
    pub(crate) hash_set: EntityHashSet,
    iter: I,
}

impl<I: Iterator<Item: TrustedEntityBorrow>> DuplicateEntityFilterIter<I> {
    pub fn new(iter: I) -> Self {
        Self {
            hash_set: EntityHashSet::default(),
            iter,
        }
    }
}

impl<I: Iterator<Item: TrustedEntityBorrow>> Iterator for DuplicateEntityFilterIter<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .by_ref()
            .filter(|e| self.hash_set.insert(e.entity()))
            .next()
    }
}

impl<I: ExactSizeIterator<Item: TrustedEntityBorrow>> ExactSizeIterator
    for DuplicateEntityFilterIter<I>
{
}

impl<I: DoubleEndedIterator<Item: TrustedEntityBorrow>> DoubleEndedIterator
    for DuplicateEntityFilterIter<I>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back()
    }
}

impl<I: FusedIterator<Item: TrustedEntityBorrow>> FusedIterator for DuplicateEntityFilterIter<I> {}

impl<I: Iterator<Item: TrustedEntityBorrow> + Clone> Clone for DuplicateEntityFilterIter<I> {
    fn clone(&self) -> Self {
        Self {
            hash_set: self.hash_set.clone(),
            iter: self.iter.clone(),
        }
    }
}

impl<I: Iterator<Item: TrustedEntityBorrow> + Default> Default for DuplicateEntityFilterIter<I> {
    fn default() -> Self {
        Self {
            hash_set: Default::default(),
            iter: Default::default(),
        }
    }
}

impl<I: Iterator<Item: TrustedEntityBorrow> + Debug> Debug for DuplicateEntityFilterIter<I> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuplicateEntityFilterIter")
            .field("hash_set", &self.hash_set)
            .field("iter", &self.iter)
            .finish()
    }
}

unsafe impl<I: Iterator<Item: TrustedEntityBorrow>> EntitySet for DuplicateEntityFilterIter<I> {}

/// An iterator that yields unique entity/entity borrow slices.
pub struct UniqueEntitySliceIter<'a, T: TrustedEntityBorrow + 'a, I: Iterator<Item = &'a [T]>> {
    pub(crate) iter: I,
}

impl<'a, T: TrustedEntityBorrow + 'a, I: Iterator<Item = &'a [T]> + AsRef<[&'a [T]]>>
    UniqueEntitySliceIter<'a, T, I>
{
    pub unsafe fn from_slice_iterator_unchecked(iter: I) -> Self {
        Self { iter }
    }
}

impl<'a, T: TrustedEntityBorrow> UniqueEntitySliceIter<'a, T, slice::ChunksExact<'a, T>> {
    pub fn remainder(&self) -> &'a UniqueEntitySlice<T> {
        // SAFETY: All elements in the original iterator are unique slices.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.iter.remainder()) }
    }
}

impl<'a, T: TrustedEntityBorrow> UniqueEntitySliceIter<'a, T, slice::RChunksExact<'a, T>> {
    pub fn remainder(&self) -> &'a UniqueEntitySlice<T> {
        // SAFETY: All elements in the original iterator are unique slices.
        unsafe { UniqueEntitySlice::from_slice_unchecked(self.iter.remainder()) }
    }
}

impl<'a, T: TrustedEntityBorrow + 'a, I: Iterator<Item = &'a [T]>> Iterator
    for UniqueEntitySliceIter<'a, T, I>
{
    type Item = &'a UniqueEntitySlice<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|slice|
        // SAFETY: All elements in the original iterator are unique slices.
        unsafe { UniqueEntitySlice::from_slice_unchecked(slice) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T: TrustedEntityBorrow + 'a, I: ExactSizeIterator<Item = &'a [T]>> ExactSizeIterator
    for UniqueEntitySliceIter<'a, T, I>
{
}

impl<'a, T: TrustedEntityBorrow + 'a, I: DoubleEndedIterator<Item = &'a [T]>> DoubleEndedIterator
    for UniqueEntitySliceIter<'a, T, I>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|slice|
            // SAFETY: All elements in the original iterator are unique slices.
            unsafe { UniqueEntitySlice::from_slice_unchecked(slice) })
    }
}

impl<'a, T: TrustedEntityBorrow + 'a, I: FusedIterator<Item = &'a [T]>> FusedIterator
    for UniqueEntitySliceIter<'a, T, I>
{
}

impl<'a, T: TrustedEntityBorrow + 'a, I: Iterator<Item = &'a [T]> + AsRef<[&'a [T]]>>
    AsRef<[&'a UniqueEntitySlice<T>]> for UniqueEntitySliceIter<'a, T, I>
{
    fn as_ref(&self) -> &[&'a UniqueEntitySlice<T>] {
        // SAFETY:
        unsafe { UniqueEntitySlice::cast_slice_of_unique_entity_slices(self.iter.as_ref()) }
    }
}

impl<'a, T: TrustedEntityBorrow + 'a, I: Iterator<Item = &'a [T]> + Default> Default
    for UniqueEntitySliceIter<'a, T, I>
{
    fn default() -> Self {
        Self {
            iter: Default::default(),
        }
    }
}

impl<'a, T: TrustedEntityBorrow + 'a, I: Iterator<Item = &'a [T]> + Clone> Clone
    for UniqueEntitySliceIter<'a, T, I>
{
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
        }
    }
}

impl<'a, T: TrustedEntityBorrow + 'a, I: Iterator<Item = &'a [T]>> Debug
    for UniqueEntitySliceIter<'a, T, I>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("UniqueEntitySliceIter").finish()
    }
}

pub type Windows<'a, T> = UniqueEntitySliceIter<'a, T, slice::Windows<'a, T>>;

pub type Chunks<'a, T> = UniqueEntitySliceIter<'a, T, slice::Chunks<'a, T>>;

pub type ChunksExact<'a, T> = UniqueEntitySliceIter<'a, T, slice::ChunksExact<'a, T>>;

pub type RChunks<'a, T> = UniqueEntitySliceIter<'a, T, slice::RChunks<'a, T>>;

pub type RChunksExact<'a, T> = UniqueEntitySliceIter<'a, T, slice::RChunksExact<'a, T>>;

pub type ChunkBy<'a, T, P> = UniqueEntitySliceIter<'a, T, slice::ChunkBy<'a, T, P>>;

pub type Split<'a, T, P> = UniqueEntitySliceIter<'a, T, slice::Split<'a, T, P>>;

pub type SplitInclusive<'a, T, P> = UniqueEntitySliceIter<'a, T, slice::SplitInclusive<'a, T, P>>;

pub type RSplit<'a, T, P> = UniqueEntitySliceIter<'a, T, slice::RSplit<'a, T, P>>;

pub type SplitN<'a, T, P> = UniqueEntitySliceIter<'a, T, slice::SplitN<'a, T, P>>;

pub type RSplitN<'a, T, P> = UniqueEntitySliceIter<'a, T, slice::RSplitN<'a, T, P>>;

/// An iterator that yields unique, mutable entity/entity borrow slices.
pub struct UniqueEntitySliceIterMut<
    'a,
    T: TrustedEntityBorrow + 'a,
    I: Iterator<Item = &'a mut [T]>,
> {
    pub(crate) iter: I,
}

impl<'a, T: TrustedEntityBorrow + 'a, I: Iterator<Item = &'a mut [T]>>
    UniqueEntitySliceIterMut<'a, T, I>
{
    pub unsafe fn from_mut_slice_iterator_unchecked(iter: I) -> Self {
        Self { iter }
    }
}

impl<'a, T: TrustedEntityBorrow> UniqueEntitySliceIterMut<'a, T, slice::ChunksExactMut<'a, T>> {
    pub fn into_remainder(self) -> &'a mut UniqueEntitySlice<T> {
        // SAFETY: All elements in the original iterator are unique slices.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.iter.into_remainder()) }
    }
}

impl<'a, T: TrustedEntityBorrow> UniqueEntitySliceIterMut<'a, T, slice::RChunksExactMut<'a, T>> {
    pub fn into_remainder(self) -> &'a mut UniqueEntitySlice<T> {
        // SAFETY: All elements in the original iterator are unique slices.
        unsafe { UniqueEntitySlice::from_slice_unchecked_mut(self.iter.into_remainder()) }
    }
}

impl<'a, T: TrustedEntityBorrow + 'a, I: Iterator<Item = &'a mut [T]>> Iterator
    for UniqueEntitySliceIterMut<'a, T, I>
{
    type Item = &'a mut UniqueEntitySlice<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|slice|
            // SAFETY: All elements in the original iterator are unique slices.
            unsafe { UniqueEntitySlice::from_slice_unchecked_mut(slice) })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T: TrustedEntityBorrow + 'a, I: ExactSizeIterator<Item = &'a mut [T]>> ExactSizeIterator
    for UniqueEntitySliceIterMut<'a, T, I>
{
}

impl<'a, T: TrustedEntityBorrow + 'a, I: DoubleEndedIterator<Item = &'a mut [T]>>
    DoubleEndedIterator for UniqueEntitySliceIterMut<'a, T, I>
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|slice|
            // SAFETY: All elements in the original iterator are unique slices.
            unsafe { UniqueEntitySlice::from_slice_unchecked_mut(slice) })
    }
}

impl<'a, T: TrustedEntityBorrow + 'a, I: FusedIterator<Item = &'a mut [T]>> FusedIterator
    for UniqueEntitySliceIterMut<'a, T, I>
{
}

impl<'a, T: TrustedEntityBorrow + 'a, I: Iterator<Item = &'a mut [T]> + AsRef<[&'a [T]]>>
    AsRef<[&'a UniqueEntitySlice<T>]> for UniqueEntitySliceIterMut<'a, T, I>
{
    fn as_ref(&self) -> &[&'a UniqueEntitySlice<T>] {
        // SAFETY: All elements in the original iterator are unique slices.
        unsafe { UniqueEntitySlice::cast_slice_of_unique_entity_slices(self.iter.as_ref()) }
    }
}

impl<'a, T: TrustedEntityBorrow + 'a, I: Iterator<Item = &'a mut [T]> + AsMut<[&'a mut [T]]>>
    AsMut<[&'a mut UniqueEntitySlice<T>]> for UniqueEntitySliceIterMut<'a, T, I>
{
    fn as_mut(&mut self) -> &mut [&'a mut UniqueEntitySlice<T>] {
        // SAFETY: All elements in the original iterator are unique slices.
        unsafe { UniqueEntitySlice::cast_slice_of_unique_entity_slices_mut(self.iter.as_mut()) }
    }
}

impl<'a, T: TrustedEntityBorrow + 'a, I: Iterator<Item = &'a mut [T]> + Default> Default
    for UniqueEntitySliceIterMut<'a, T, I>
{
    fn default() -> Self {
        Self {
            iter: Default::default(),
        }
    }
}

impl<'a, T: TrustedEntityBorrow + 'a, I: Iterator<Item = &'a mut [T]> + Clone> Clone
    for UniqueEntitySliceIterMut<'a, T, I>
{
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
        }
    }
}

impl<'a, T: TrustedEntityBorrow + 'a, I: Iterator<Item = &'a mut [T]>> Debug
    for UniqueEntitySliceIterMut<'a, T, I>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("UniqueEntitySliceIterMut").finish()
    }
}

pub type ChunksMut<'a, T> = UniqueEntitySliceIterMut<'a, T, slice::ChunksMut<'a, T>>;

pub type ChunksExactMut<'a, T> = UniqueEntitySliceIterMut<'a, T, slice::ChunksExactMut<'a, T>>;

pub type RChunksMut<'a, T> = UniqueEntitySliceIterMut<'a, T, slice::RChunksMut<'a, T>>;

pub type RChunksExactMut<'a, T> = UniqueEntitySliceIterMut<'a, T, slice::RChunksExactMut<'a, T>>;

pub type ChunkByMut<'a, T, P> = UniqueEntitySliceIterMut<'a, T, slice::ChunkByMut<'a, T, P>>;

pub type SplitMut<'a, T, P> = UniqueEntitySliceIterMut<'a, T, slice::SplitMut<'a, T, P>>;

pub type SplitInclusiveMut<'a, T, P> =
    UniqueEntitySliceIterMut<'a, T, slice::SplitInclusiveMut<'a, T, P>>;

pub type RSplitMut<'a, T, P> = UniqueEntitySliceIterMut<'a, T, slice::RSplitMut<'a, T, P>>;

pub type SplitNMut<'a, T, P> = UniqueEntitySliceIterMut<'a, T, slice::SplitNMut<'a, T, P>>;

pub type RSplitNMut<'a, T, P> = UniqueEntitySliceIterMut<'a, T, slice::RSplitNMut<'a, T, P>>;
