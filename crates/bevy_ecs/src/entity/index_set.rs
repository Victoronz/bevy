use core::{
    fmt::Debug,
    iter::FusedIterator,
    marker::PhantomData,
    ops::{
        Bound, Deref, DerefMut, Index, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive,
        RangeTo, RangeToInclusive,
    },
    ptr,
};
use std::{
    cmp::Ordering,
    hash::{BuildHasher, Hash},
    ops::{BitAnd, BitOr, BitXor, Sub},
};

use indexmap::set::{self, IndexSet};

use super::{Entity, EntityHash, EntitySet, UniqueEntityVec};

#[derive(Debug, Clone, Default)]
pub struct EntityIndexSet(pub(crate) IndexSet<Entity, EntityHash>);

impl EntityIndexSet {
    pub fn new() -> Self {
        Self(IndexSet::with_hasher(EntityHash))
    }

    pub fn with_capacity(n: usize) -> Self {
        Self(IndexSet::with_capacity_and_hasher(n, EntityHash))
    }

    pub fn as_slice(&self) -> &EntityIndexSetSlice {
        // SAFETY: EntityIndexSetSlice is a transparent wrapper around indexmap::set::Slice.
        unsafe { EntityIndexSetSlice::from_slice_unchecked(self.0.as_slice()) }
    }

    pub fn drain<R: RangeBounds<usize>>(&mut self, range: R) -> Drain<'_> {
        Drain(self.0.drain(range), PhantomData)
    }

    pub fn get_range<R: RangeBounds<usize>>(&self, range: R) -> Option<&EntityIndexSetSlice> {
        self.0.get_range(range).map(|slice|
            // SAFETY: The source IndexSet used EntityHash.
            unsafe { EntityIndexSetSlice::from_slice_unchecked(slice) })
    }

    pub fn iter(&self) -> Iter<'_> {
        Iter(self.0.iter(), PhantomData)
    }

    pub fn into_boxed_slice(self) -> Box<EntityIndexSetSlice> {
        // SAFETY: EntityIndexSetSlice is a transparent wrapper around indexmap::set::Slice.
        unsafe { EntityIndexSetSlice::from_boxed_slice_unchecked(self.0.into_boxed_slice()) }
    }
}

impl Deref for EntityIndexSet {
    type Target = IndexSet<Entity, EntityHash>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for EntityIndexSet {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a> IntoIterator for &'a EntityIndexSet {
    type Item = &'a Entity;

    type IntoIter = Iter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Iter((&self.0).into_iter(), PhantomData)
    }
}

impl IntoIterator for EntityIndexSet {
    type Item = Entity;

    type IntoIter = IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter(self.0.into_iter(), PhantomData)
    }
}

impl BitAnd for &EntityIndexSet {
    type Output = EntityIndexSet;

    fn bitand(self, rhs: Self) -> Self::Output {
        EntityIndexSet(self.0.bitand(&rhs.0))
    }
}

impl BitOr for &EntityIndexSet {
    type Output = EntityIndexSet;

    fn bitor(self, rhs: Self) -> Self::Output {
        EntityIndexSet(self.0.bitor(&rhs.0))
    }
}

impl BitXor for &EntityIndexSet {
    type Output = EntityIndexSet;

    fn bitxor(self, rhs: Self) -> Self::Output {
        EntityIndexSet(self.0.bitxor(&rhs.0))
    }
}

impl Sub for &EntityIndexSet {
    type Output = EntityIndexSet;

    fn sub(self, rhs: Self) -> Self::Output {
        EntityIndexSet(self.0.sub(&rhs.0))
    }
}

impl<'a> Extend<&'a Entity> for EntityIndexSet {
    fn extend<T: IntoIterator<Item = &'a Entity>>(&mut self, iter: T) {
        self.0.extend(iter)
    }
}

impl Extend<Entity> for EntityIndexSet {
    fn extend<T: IntoIterator<Item = Entity>>(&mut self, iter: T) {
        self.0.extend(iter)
    }
}

impl<const N: usize> From<[Entity; N]> for EntityIndexSet {
    fn from(value: [Entity; N]) -> Self {
        Self(IndexSet::from_iter(value))
    }
}

impl FromIterator<Entity> for EntityIndexSet {
    fn from_iter<I: IntoIterator<Item = Entity>>(iterable: I) -> Self {
        Self(IndexSet::from_iter(iterable))
    }
}

impl<S2> PartialEq<IndexSet<Entity, S2>> for EntityIndexSet
where
    S2: BuildHasher,
{
    fn eq(&self, other: &IndexSet<Entity, S2>) -> bool {
        self.0.eq(other)
    }
}

impl PartialEq for EntityIndexSet {
    fn eq(&self, other: &EntityIndexSet) -> bool {
        self.0.eq(other)
    }
}

impl Eq for EntityIndexSet {}

impl Index<(Bound<usize>, Bound<usize>)> for EntityIndexSet {
    type Output = EntityIndexSetSlice;
    fn index(&self, key: (Bound<usize>, Bound<usize>)) -> &Self::Output {
        // SAFETY: The source IndexSet used EntityHash.
        unsafe { EntityIndexSetSlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl Index<Range<usize>> for EntityIndexSet {
    type Output = EntityIndexSetSlice;
    fn index(&self, key: Range<usize>) -> &Self::Output {
        // SAFETY: The source IndexSet used EntityHash.
        unsafe { EntityIndexSetSlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl Index<RangeFrom<usize>> for EntityIndexSet {
    type Output = EntityIndexSetSlice;
    fn index(&self, key: RangeFrom<usize>) -> &Self::Output {
        // SAFETY: The source IndexSet used EntityHash.
        unsafe { EntityIndexSetSlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl Index<RangeFull> for EntityIndexSet {
    type Output = EntityIndexSetSlice;
    fn index(&self, key: RangeFull) -> &Self::Output {
        // SAFETY: The source IndexSet used EntityHash.
        unsafe { EntityIndexSetSlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl Index<RangeInclusive<usize>> for EntityIndexSet {
    type Output = EntityIndexSetSlice;
    fn index(&self, key: RangeInclusive<usize>) -> &Self::Output {
        // SAFETY: The source IndexSet used EntityHash.
        unsafe { EntityIndexSetSlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl Index<RangeTo<usize>> for EntityIndexSet {
    type Output = EntityIndexSetSlice;
    fn index(&self, key: RangeTo<usize>) -> &Self::Output {
        // SAFETY: The source IndexSet used EntityHash.
        unsafe { EntityIndexSetSlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl Index<RangeToInclusive<usize>> for EntityIndexSet {
    type Output = EntityIndexSetSlice;
    fn index(&self, key: RangeToInclusive<usize>) -> &Self::Output {
        // SAFETY: The source IndexSet used EntityHash.
        unsafe { EntityIndexSetSlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl Index<usize> for EntityIndexSet {
    type Output = Entity;
    fn index(&self, key: usize) -> &Entity {
        self.0.index(key)
    }
}

impl From<EntityIndexSet> for UniqueEntityVec<Entity> {
    fn from(value: EntityIndexSet) -> Self {
        // SAFETY: All elements in the source set are unique.
        unsafe { Self::from_vec_unchecked(value.into_iter().collect::<Vec<Entity>>()) }
    }
}

#[repr(transparent)]
pub struct EntityIndexSetSlice<S = EntityHash>(PhantomData<S>, set::Slice<Entity>);

impl EntityIndexSetSlice {
    pub const fn new<'a>() -> &'a Self {
        unsafe { Self::from_slice_unchecked(set::Slice::new()) }
    }

    pub const unsafe fn from_slice_unchecked(slice: &set::Slice<Entity>) -> &Self {
        // SAFETY: EntityIndexSetSlice is a transparent wrapper around indexmap::set::Slice.
        unsafe { &*(ptr::from_ref(slice) as *const Self) }
    }

    pub const unsafe fn from_slice_unchecked_mut(slice: &mut set::Slice<Entity>) -> &mut Self {
        // SAFETY: EntityIndexSetSlice is a transparent wrapper around indexmap::set::Slice.
        unsafe { &mut *(ptr::from_mut(slice) as *mut Self) }
    }

    pub unsafe fn from_boxed_slice_unchecked(slice: Box<set::Slice<Entity>>) -> Box<Self> {
        // SAFETY: EntityIndexSetSlice is a transparent wrapper around indexmap::map::Slice.
        unsafe { Box::from_raw(Box::into_raw(slice) as *mut Self) }
    }

    pub fn into_boxed_inner(self: Box<Self>) -> Box<set::Slice<Entity>> {
        // SAFETY: EntityIndexSetSlice is a transparent wrapper around indexmap::map::Slice.
        unsafe { Box::from_raw(Box::into_raw(self) as *mut set::Slice<Entity>) }
    }

    pub fn get_range<R: RangeBounds<usize>>(&self, range: R) -> Option<&Self> {
        self.1.get_range(range).map(|slice|
            // SAFETY: The source IndexSet used EntityHash.
            unsafe { Self::from_slice_unchecked(slice) })
    }

    pub fn split_at(&self, index: usize) -> (&Self, &Self) {
        let (slice_1, slice_2) = self.1.split_at(index);
        // SAFETY: The source IndexSet used EntityHash.
        unsafe {
            (
                Self::from_slice_unchecked(slice_1),
                Self::from_slice_unchecked(slice_2),
            )
        }
    }

    pub fn split_first(&self) -> Option<(&Entity, &Self)> {
        self.1
            .split_first()
            .map(|(first, rest)| (first, unsafe { Self::from_slice_unchecked(rest) }))
    }

    pub fn split_last(&self) -> Option<(&Entity, &Self)> {
        self.1
            .split_last()
            .map(|(last, rest)| (last, unsafe { Self::from_slice_unchecked(rest) }))
    }

    pub fn iter(&self) -> Iter<'_> {
        Iter(self.1.iter(), PhantomData)
    }
}

impl Deref for EntityIndexSetSlice {
    type Target = set::Slice<Entity>;

    fn deref(&self) -> &Self::Target {
        &self.1
    }
}

impl Clone for Box<EntityIndexSetSlice> {
    fn clone(&self) -> Self {
        unsafe {
            let slice = &*(ptr::from_ref(&self.1) as *const Box<set::Slice<Entity>>);
            EntityIndexSetSlice::from_boxed_slice_unchecked(slice.clone())
        }
    }
}

impl Default for &EntityIndexSetSlice {
    fn default() -> Self {
        unsafe { EntityIndexSetSlice::from_slice_unchecked(<&set::Slice<Entity>>::default()) }
    }
}

impl Default for Box<EntityIndexSetSlice> {
    fn default() -> Self {
        unsafe {
            EntityIndexSetSlice::from_boxed_slice_unchecked(<Box<set::Slice<Entity>>>::default())
        }
    }
}

impl From<&EntityIndexSetSlice> for Box<EntityIndexSetSlice> {
    fn from(value: &EntityIndexSetSlice) -> Self {
        unsafe { EntityIndexSetSlice::from_boxed_slice_unchecked(value.1.into()) }
    }
}

impl Hash for EntityIndexSetSlice {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.1.hash(state);
    }
}

impl PartialOrd for EntityIndexSetSlice {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.1.partial_cmp(&other.1)
    }
}

impl Ord for EntityIndexSetSlice {
    fn cmp(&self, other: &Self) -> Ordering {
        self.1.cmp(other)
    }
}

impl PartialEq for EntityIndexSetSlice {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl Eq for EntityIndexSetSlice {}

impl Index<(Bound<usize>, Bound<usize>)> for EntityIndexSetSlice {
    type Output = Self;
    fn index(&self, key: (Bound<usize>, Bound<usize>)) -> &Self {
        // SAFETY: The source IndexSet used EntityHash.
        unsafe { Self::from_slice_unchecked(self.1.index(key)) }
    }
}

impl Index<Range<usize>> for EntityIndexSetSlice {
    type Output = Self;
    fn index(&self, key: Range<usize>) -> &Self {
        // SAFETY: The source IndexSet used EntityHash.
        unsafe { Self::from_slice_unchecked(self.1.index(key)) }
    }
}

impl Index<RangeFrom<usize>> for EntityIndexSetSlice {
    type Output = EntityIndexSetSlice;
    fn index(&self, key: RangeFrom<usize>) -> &Self {
        // SAFETY: The source IndexSet used EntityHash.
        unsafe { Self::from_slice_unchecked(self.1.index(key)) }
    }
}

impl Index<RangeFull> for EntityIndexSetSlice {
    type Output = Self;
    fn index(&self, key: RangeFull) -> &Self {
        // SAFETY: The source IndexSet used EntityHash.
        unsafe { Self::from_slice_unchecked(self.1.index(key)) }
    }
}

impl Index<RangeInclusive<usize>> for EntityIndexSetSlice {
    type Output = Self;
    fn index(&self, key: RangeInclusive<usize>) -> &Self {
        // SAFETY: The source IndexSet used EntityHash.
        unsafe { Self::from_slice_unchecked(self.1.index(key)) }
    }
}

impl Index<RangeTo<usize>> for EntityIndexSetSlice {
    type Output = Self;
    fn index(&self, key: RangeTo<usize>) -> &Self {
        // SAFETY: The source IndexSet used EntityHash.
        unsafe { Self::from_slice_unchecked(self.1.index(key)) }
    }
}

impl Index<RangeToInclusive<usize>> for EntityIndexSetSlice {
    type Output = Self;
    fn index(&self, key: RangeToInclusive<usize>) -> &Self {
        // SAFETY: The source IndexSet used EntityHash.
        unsafe { Self::from_slice_unchecked(self.1.index(key)) }
    }
}

impl Index<usize> for EntityIndexSetSlice {
    type Output = Entity;
    fn index(&self, key: usize) -> &Entity {
        self.1.index(key)
    }
}

pub struct Iter<'a, S = EntityHash>(set::Iter<'a, Entity>, PhantomData<S>);

impl Iter<'_> {
    pub fn as_slice(&self) -> &EntityIndexSetSlice {
        // SAFETY: The source IndexSet used EntityHash.
        unsafe { EntityIndexSetSlice::from_slice_unchecked(self.0.as_slice()) }
    }
}

impl<'a> Deref for Iter<'a> {
    type Target = set::Iter<'a, Entity>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = &'a Entity;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl DoubleEndedIterator for Iter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl ExactSizeIterator for Iter<'_> {}

impl FusedIterator for Iter<'_> {}

impl Clone for Iter<'_> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl Debug for Iter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Iter").field(&self.0).field(&self.1).finish()
    }
}

impl Default for Iter<'_> {
    fn default() -> Self {
        Self(Default::default(), PhantomData)
    }
}

unsafe impl EntitySet for Iter<'_> {}

pub struct IntoIter<S = EntityHash>(set::IntoIter<Entity>, PhantomData<S>);

impl IntoIter {
    pub fn as_slice(&self) -> &EntityIndexSetSlice {
        // SAFETY: The source IndexSet used EntityHash.
        unsafe { EntityIndexSetSlice::from_slice_unchecked(self.0.as_slice()) }
    }
}

impl Iterator for IntoIter {
    type Item = Entity;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl DoubleEndedIterator for IntoIter {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl ExactSizeIterator for IntoIter {}

impl FusedIterator for IntoIter {}

impl Clone for IntoIter {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl Debug for IntoIter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("IntoIter")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

impl Default for IntoIter {
    fn default() -> Self {
        Self(Default::default(), PhantomData)
    }
}

unsafe impl EntitySet for IntoIter {}

pub struct Drain<'a, S = EntityHash>(set::Drain<'a, Entity>, PhantomData<S>);

impl Drain<'_> {
    pub fn as_slice(&self) -> &EntityIndexSetSlice {
        // SAFETY: EntityIndexSetSlice is a transparent wrapper around indexmap::set::Slice.
        unsafe { EntityIndexSetSlice::from_slice_unchecked(self.0.as_slice()) }
    }
}

impl<'a> Iterator for Drain<'a> {
    type Item = Entity;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl DoubleEndedIterator for Drain<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl ExactSizeIterator for Drain<'_> {}

impl FusedIterator for Drain<'_> {}

impl Debug for Drain<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Drain")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

unsafe impl EntitySet for Drain<'_> {}

unsafe impl EntitySet for set::Difference<'_, Entity, EntityHash> {}

unsafe impl EntitySet for set::Intersection<'_, Entity, EntityHash> {}

unsafe impl EntitySet for set::SymmetricDifference<'_, Entity, EntityHash, EntityHash> {}

unsafe impl EntitySet for set::Union<'_, Entity, EntityHash> {}

unsafe impl<I: Iterator<Item = Entity>> EntitySet for set::Splice<'_, I, Entity, EntityHash> {}
