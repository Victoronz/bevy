use core::{
    marker::PhantomData,
    ops::{
        Bound, Deref, DerefMut, Index, IndexMut, Range, RangeBounds, RangeFrom, RangeFull,
        RangeInclusive, RangeTo, RangeToInclusive,
    },
    ptr,
};

use alloc::fmt::Debug;
use std::{
    cmp::Ordering,
    hash::{BuildHasher, Hash},
    iter::FusedIterator,
};

use indexmap::map::{self, IndexMap, IntoValues, ValuesMut};

use super::{Entity, EntityHash, EntitySet, TrustedEntityBorrow};

#[derive(Debug, Clone)]
pub struct EntityIndexMap<V>(pub(crate) IndexMap<Entity, V, EntityHash>);

impl<V> EntityIndexMap<V> {
    pub fn new() -> Self {
        Self(IndexMap::with_hasher(EntityHash))
    }

    pub fn with_capacity(n: usize) -> Self {
        Self(IndexMap::with_capacity_and_hasher(n, EntityHash))
    }

    pub fn as_slice(&self) -> &EntityIndexMapSlice<V> {
        // SAFETY: EntityIndexMapSlice is a transparent wrapper around indexmap::map::Slice.
        unsafe { EntityIndexMapSlice::from_slice_unchecked(self.0.as_slice()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut EntityIndexMapSlice<V> {
        // SAFETY: EntityIndexMapSlice is a transparent wrapper around indexmap::map::Slice.
        unsafe { EntityIndexMapSlice::from_slice_unchecked_mut(self.0.as_mut_slice()) }
    }

    pub fn into_boxed_slice(self) -> Box<EntityIndexMapSlice<V>> {
        // SAFETY: EntityIndexMapSlice is a transparent wrapper around indexmap::map::Slice.
        unsafe { EntityIndexMapSlice::from_boxed_slice_unchecked(self.0.into_boxed_slice()) }
    }

    pub fn get_range<R: RangeBounds<usize>>(&self, range: R) -> Option<&EntityIndexMapSlice<V>> {
        self.0.get_range(range).map(|slice|
            // SAFETY: EntityIndexSetSlice is a transparent wrapper around indexmap::set::Slice.
            unsafe { EntityIndexMapSlice::from_slice_unchecked(slice) })
    }

    pub fn get_range_mut<R: RangeBounds<usize>>(
        &mut self,
        range: R,
    ) -> Option<&mut EntityIndexMapSlice<V>> {
        self.0.get_range_mut(range).map(|slice|
            // SAFETY: EntityIndexSetSlice is a transparent wrapper around indexmap::set::Slice.
            unsafe { EntityIndexMapSlice::from_slice_unchecked_mut(slice) })
    }

    pub fn iter(&self) -> Iter<'_, V> {
        Iter(self.0.iter(), PhantomData)
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, V> {
        IterMut(self.0.iter_mut(), PhantomData)
    }

    pub fn drain<R: RangeBounds<usize>>(&mut self, range: R) -> Drain<'_, V> {
        Drain(self.0.drain(range), PhantomData)
    }

    pub fn keys(&self) -> Keys<'_, V> {
        Keys(self.0.keys(), PhantomData)
    }

    pub fn into_keys(self) -> IntoKeys<V> {
        IntoKeys(self.0.into_keys(), PhantomData)
    }
}

impl<V> Default for EntityIndexMap<V> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<V> Deref for EntityIndexMap<V> {
    type Target = IndexMap<Entity, V, EntityHash>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<V> DerefMut for EntityIndexMap<V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, V: Copy> Extend<(&'a Entity, &'a V)> for EntityIndexMap<V> {
    fn extend<T: IntoIterator<Item = (&'a Entity, &'a V)>>(&mut self, iter: T) {
        self.0.extend(iter)
    }
}

impl<V> Extend<(Entity, V)> for EntityIndexMap<V> {
    fn extend<T: IntoIterator<Item = (Entity, V)>>(&mut self, iter: T) {
        self.0.extend(iter)
    }
}

impl<V, const N: usize> From<[(Entity, V); N]> for EntityIndexMap<V> {
    fn from(value: [(Entity, V); N]) -> Self {
        Self(IndexMap::from_iter(value))
    }
}

impl<V> FromIterator<(Entity, V)> for EntityIndexMap<V> {
    fn from_iter<I: IntoIterator<Item = (Entity, V)>>(iterable: I) -> Self {
        Self(IndexMap::from_iter(iterable))
    }
}

impl<V, Q: TrustedEntityBorrow + ?Sized> Index<&Q> for EntityIndexMap<V> {
    type Output = V;
    fn index(&self, key: &Q) -> &V {
        self.0.index(&key.entity())
    }
}

impl<V> Index<(Bound<usize>, Bound<usize>)> for EntityIndexMap<V> {
    type Output = EntityIndexMapSlice<V>;
    fn index(&self, key: (Bound<usize>, Bound<usize>)) -> &Self::Output {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { EntityIndexMapSlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<V> Index<Range<usize>> for EntityIndexMap<V> {
    type Output = EntityIndexMapSlice<V>;
    fn index(&self, key: Range<usize>) -> &Self::Output {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { EntityIndexMapSlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<V> Index<RangeFrom<usize>> for EntityIndexMap<V> {
    type Output = EntityIndexMapSlice<V>;
    fn index(&self, key: RangeFrom<usize>) -> &Self::Output {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { EntityIndexMapSlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<V> Index<RangeFull> for EntityIndexMap<V> {
    type Output = EntityIndexMapSlice<V>;
    fn index(&self, key: RangeFull) -> &Self::Output {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { EntityIndexMapSlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<V> Index<RangeInclusive<usize>> for EntityIndexMap<V> {
    type Output = EntityIndexMapSlice<V>;
    fn index(&self, key: RangeInclusive<usize>) -> &Self::Output {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { EntityIndexMapSlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<V> Index<RangeTo<usize>> for EntityIndexMap<V> {
    type Output = EntityIndexMapSlice<V>;
    fn index(&self, key: RangeTo<usize>) -> &Self::Output {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { EntityIndexMapSlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<V> Index<RangeToInclusive<usize>> for EntityIndexMap<V> {
    type Output = EntityIndexMapSlice<V>;
    fn index(&self, key: RangeToInclusive<usize>) -> &Self::Output {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { EntityIndexMapSlice::from_slice_unchecked(self.0.index(key)) }
    }
}

impl<V> Index<usize> for EntityIndexMap<V> {
    type Output = V;
    fn index(&self, key: usize) -> &V {
        self.0.index(key)
    }
}

impl<V, Q: TrustedEntityBorrow + ?Sized> IndexMut<&Q> for EntityIndexMap<V> {
    fn index_mut(&mut self, key: &Q) -> &mut V {
        self.0.index_mut(&key.entity())
    }
}

impl<V> IndexMut<(Bound<usize>, Bound<usize>)> for EntityIndexMap<V> {
    fn index_mut(&mut self, key: (Bound<usize>, Bound<usize>)) -> &mut Self::Output {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { EntityIndexMapSlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<V> IndexMut<Range<usize>> for EntityIndexMap<V> {
    fn index_mut(&mut self, key: Range<usize>) -> &mut Self::Output {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { EntityIndexMapSlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<V> IndexMut<RangeFrom<usize>> for EntityIndexMap<V> {
    fn index_mut(&mut self, key: RangeFrom<usize>) -> &mut Self::Output {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { EntityIndexMapSlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<V> IndexMut<RangeFull> for EntityIndexMap<V> {
    fn index_mut(&mut self, key: RangeFull) -> &mut Self::Output {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { EntityIndexMapSlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<V> IndexMut<RangeInclusive<usize>> for EntityIndexMap<V> {
    fn index_mut(&mut self, key: RangeInclusive<usize>) -> &mut Self::Output {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { EntityIndexMapSlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<V> IndexMut<RangeTo<usize>> for EntityIndexMap<V> {
    fn index_mut(&mut self, key: RangeTo<usize>) -> &mut Self::Output {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { EntityIndexMapSlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<V> IndexMut<RangeToInclusive<usize>> for EntityIndexMap<V> {
    fn index_mut(&mut self, key: RangeToInclusive<usize>) -> &mut Self::Output {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { EntityIndexMapSlice::from_slice_unchecked_mut(self.0.index_mut(key)) }
    }
}

impl<V> IndexMut<usize> for EntityIndexMap<V> {
    fn index_mut(&mut self, key: usize) -> &mut V {
        self.0.index_mut(key)
    }
}

impl<'a, V> IntoIterator for &'a EntityIndexMap<V> {
    type Item = (&'a Entity, &'a V);
    type IntoIter = Iter<'a, V>;

    fn into_iter(self) -> Self::IntoIter {
        Iter(self.0.iter(), PhantomData)
    }
}

impl<'a, V> IntoIterator for &'a mut EntityIndexMap<V> {
    type Item = (&'a Entity, &'a mut V);
    type IntoIter = IterMut<'a, V>;

    fn into_iter(self) -> Self::IntoIter {
        IterMut(self.0.iter_mut(), PhantomData)
    }
}

impl<V> IntoIterator for EntityIndexMap<V> {
    type Item = (Entity, V);
    type IntoIter = IntoIter<V>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter(self.0.into_iter(), PhantomData)
    }
}

impl<V1, V2, S2> PartialEq<IndexMap<Entity, V2, S2>> for EntityIndexMap<V1>
where
    V1: PartialEq<V2>,
    S2: BuildHasher,
{
    fn eq(&self, other: &IndexMap<Entity, V2, S2>) -> bool {
        self.0.eq(other)
    }
}

impl<V1, V2> PartialEq<EntityIndexMap<V2>> for EntityIndexMap<V1>
where
    V1: PartialEq<V2>,
{
    fn eq(&self, other: &EntityIndexMap<V2>) -> bool {
        self.0.eq(other)
    }
}

impl<V: Eq> Eq for EntityIndexMap<V> {}

pub struct EntityIndexMapSlice<V, S = EntityHash>(PhantomData<S>, map::Slice<Entity, V>);

impl<V> EntityIndexMapSlice<V> {
    pub const fn new<'a>() -> &'a Self {
        unsafe { Self::from_slice_unchecked(map::Slice::new()) }
    }

    pub fn new_mut<'a>() -> &'a mut Self {
        unsafe { Self::from_slice_unchecked_mut(map::Slice::new_mut()) }
    }

    pub const unsafe fn from_slice_unchecked(slice: &map::Slice<Entity, V>) -> &Self {
        // SAFETY: EntityIndexMapSlice is a transparent wrapper around indexmap::map::Slice.
        unsafe { &*(ptr::from_ref(slice) as *const Self) }
    }

    pub const unsafe fn from_slice_unchecked_mut(slice: &mut map::Slice<Entity, V>) -> &mut Self {
        // SAFETY: EntityIndexMapSlice is a transparent wrapper around indexmap::map::Slice.
        unsafe { &mut *(ptr::from_mut(slice) as *mut Self) }
    }

    pub unsafe fn from_boxed_slice_unchecked(slice: Box<map::Slice<Entity, V>>) -> Box<Self> {
        // SAFETY: EntityIndexMapSlice is a transparent wrapper around indexmap::map::Slice.
        unsafe { Box::from_raw(Box::into_raw(slice) as *mut Self) }
    }

    pub fn into_boxed_inner(self: Box<Self>) -> Box<map::Slice<Entity, V>> {
        // SAFETY: EntityIndexMapSlice is a transparent wrapper around indexmap::map::Slice.
        unsafe { Box::from_raw(Box::into_raw(self) as *mut map::Slice<Entity, V>) }
    }

    pub fn get_index_mut(&mut self, index: usize) -> Option<(&Entity, &mut V)> {
        self.1.get_index_mut(index)
    }

    pub fn get_range<R: RangeBounds<usize>>(&self, range: R) -> Option<&Self> {
        self.1.get_range(range).map(|slice|
            // SAFETY: EntityIndexSetSlice is a transparent wrapper around indexmap::set::Slice.
            unsafe { Self::from_slice_unchecked(slice) })
    }

    pub fn get_range_mut<R: RangeBounds<usize>>(&mut self, range: R) -> Option<&mut Self> {
        self.1.get_range_mut(range).map(|slice|
            // SAFETY: EntityIndexSetSlice is a transparent wrapper around indexmap::set::Slice.
            unsafe { Self::from_slice_unchecked_mut(slice) })
    }

    pub fn first_mut(&mut self) -> Option<(&Entity, &mut V)> {
        self.1.first_mut()
    }

    pub fn last_mut(&mut self) -> Option<(&Entity, &mut V)> {
        self.1.last_mut()
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

    pub fn split_at_mut(&mut self, index: usize) -> (&mut Self, &mut Self) {
        let (slice_1, slice_2) = self.1.split_at_mut(index);
        // SAFETY: The source IndexSet used EntityHash.
        unsafe {
            (
                Self::from_slice_unchecked_mut(slice_1),
                Self::from_slice_unchecked_mut(slice_2),
            )
        }
    }

    pub fn split_first(&self) -> Option<((&Entity, &V), &Self)> {
        self.1
            .split_first()
            .map(|(first, rest)| (first, unsafe { Self::from_slice_unchecked(rest) }))
    }

    pub fn split_first_mut(&mut self) -> Option<((&Entity, &mut V), &mut Self)> {
        self.1
            .split_first_mut()
            .map(|(first, rest)| (first, unsafe { Self::from_slice_unchecked_mut(rest) }))
    }

    pub fn split_last(&self) -> Option<((&Entity, &V), &Self)> {
        self.1
            .split_last()
            .map(|(last, rest)| (last, unsafe { Self::from_slice_unchecked(rest) }))
    }

    pub fn split_last_mut(&mut self) -> Option<((&Entity, &mut V), &mut Self)> {
        self.1
            .split_last_mut()
            .map(|(last, rest)| (last, unsafe { Self::from_slice_unchecked_mut(rest) }))
    }

    pub fn iter(&self) -> Iter<'_, V> {
        Iter(self.1.iter(), PhantomData)
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, V> {
        IterMut(self.1.iter_mut(), PhantomData)
    }

    pub fn keys(&self) -> Keys<'_, V> {
        Keys(self.1.keys(), PhantomData)
    }

    pub fn into_keys(self: Box<Self>) -> IntoKeys<V> {
        IntoKeys(self.into_boxed_inner().into_keys(), PhantomData)
    }

    pub fn values_mut(&mut self) -> ValuesMut<'_, Entity, V> {
        self.1.values_mut()
    }

    pub fn into_values(self: Box<Self>) -> IntoValues<Entity, V> {
        self.into_boxed_inner().into_values()
    }
}

impl<V> Deref for EntityIndexMapSlice<V> {
    type Target = map::Slice<Entity, V>;

    fn deref(&self) -> &Self::Target {
        &self.1
    }
}

impl<V: Debug> Debug for EntityIndexMapSlice<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("EntityIndexMapSlice")
            .field(&self.0)
            .field(&&self.1)
            .finish()
    }
}

impl<V: Clone> Clone for Box<EntityIndexMapSlice<V>> {
    fn clone(&self) -> Self {
        unsafe {
            let slice = &*(ptr::from_ref(&self.1) as *const Box<map::Slice<Entity, V>>);
            EntityIndexMapSlice::from_boxed_slice_unchecked(slice.clone())
        }
    }
}

impl<V> Default for &EntityIndexMapSlice<V> {
    fn default() -> Self {
        unsafe { EntityIndexMapSlice::from_slice_unchecked(<&map::Slice<Entity, V>>::default()) }
    }
}

impl<V> Default for &mut EntityIndexMapSlice<V> {
    fn default() -> Self {
        unsafe {
            EntityIndexMapSlice::from_slice_unchecked_mut(<&mut map::Slice<Entity, V>>::default())
        }
    }
}

impl<V> Default for Box<EntityIndexMapSlice<V>> {
    fn default() -> Self {
        unsafe {
            EntityIndexMapSlice::from_boxed_slice_unchecked(<Box<map::Slice<Entity, V>>>::default())
        }
    }
}

impl<V: Copy> From<&EntityIndexMapSlice<V>> for Box<EntityIndexMapSlice<V>> {
    fn from(value: &EntityIndexMapSlice<V>) -> Self {
        unsafe { EntityIndexMapSlice::from_boxed_slice_unchecked(value.1.into()) }
    }
}

impl<V: Hash> Hash for EntityIndexMapSlice<V> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.1.hash(state);
    }
}

impl<'a, V> IntoIterator for &'a EntityIndexMapSlice<V> {
    type Item = (&'a Entity, &'a V);
    type IntoIter = Iter<'a, V>;

    fn into_iter(self) -> Self::IntoIter {
        Iter(self.1.iter(), PhantomData)
    }
}

impl<'a, V> IntoIterator for &'a mut EntityIndexMapSlice<V> {
    type Item = (&'a Entity, &'a mut V);
    type IntoIter = IterMut<'a, V>;

    fn into_iter(self) -> Self::IntoIter {
        IterMut(self.1.iter_mut(), PhantomData)
    }
}

impl<V> IntoIterator for Box<EntityIndexMapSlice<V>> {
    type Item = (Entity, V);
    type IntoIter = IntoIter<V>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter(self.into_boxed_inner().into_iter(), PhantomData)
    }
}

impl<V: PartialOrd> PartialOrd for EntityIndexMapSlice<V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.1.partial_cmp(&other.1)
    }
}

impl<V: Ord> Ord for EntityIndexMapSlice<V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.1.cmp(other)
    }
}

impl<V: PartialEq> PartialEq for EntityIndexMapSlice<V> {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1
    }
}

impl<V: Eq> Eq for EntityIndexMapSlice<V> {}

impl<V> Index<(Bound<usize>, Bound<usize>)> for EntityIndexMapSlice<V> {
    type Output = Self;
    fn index(&self, key: (Bound<usize>, Bound<usize>)) -> &Self {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { Self::from_slice_unchecked(self.1.index(key)) }
    }
}

impl<V> Index<Range<usize>> for EntityIndexMapSlice<V> {
    type Output = Self;
    fn index(&self, key: Range<usize>) -> &Self {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { Self::from_slice_unchecked(self.1.index(key)) }
    }
}

impl<V> Index<RangeFrom<usize>> for EntityIndexMapSlice<V> {
    type Output = Self;
    fn index(&self, key: RangeFrom<usize>) -> &Self {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { Self::from_slice_unchecked(self.1.index(key)) }
    }
}

impl<V> Index<RangeFull> for EntityIndexMapSlice<V> {
    type Output = Self;
    fn index(&self, key: RangeFull) -> &Self {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { Self::from_slice_unchecked(self.1.index(key)) }
    }
}

impl<V> Index<RangeInclusive<usize>> for EntityIndexMapSlice<V> {
    type Output = Self;
    fn index(&self, key: RangeInclusive<usize>) -> &Self {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { Self::from_slice_unchecked(self.1.index(key)) }
    }
}

impl<V> Index<RangeTo<usize>> for EntityIndexMapSlice<V> {
    type Output = Self;
    fn index(&self, key: RangeTo<usize>) -> &Self {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { Self::from_slice_unchecked(self.1.index(key)) }
    }
}

impl<V> Index<RangeToInclusive<usize>> for EntityIndexMapSlice<V> {
    type Output = Self;
    fn index(&self, key: RangeToInclusive<usize>) -> &Self {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { Self::from_slice_unchecked(self.1.index(key)) }
    }
}

impl<V> Index<usize> for EntityIndexMapSlice<V> {
    type Output = V;
    fn index(&self, key: usize) -> &V {
        self.1.index(key)
    }
}

impl<V> IndexMut<(Bound<usize>, Bound<usize>)> for EntityIndexMapSlice<V> {
    fn index_mut(&mut self, key: (Bound<usize>, Bound<usize>)) -> &mut Self {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { Self::from_slice_unchecked_mut(self.1.index_mut(key)) }
    }
}

impl<V> IndexMut<Range<usize>> for EntityIndexMapSlice<V> {
    fn index_mut(&mut self, key: Range<usize>) -> &mut Self {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { Self::from_slice_unchecked_mut(self.1.index_mut(key)) }
    }
}

impl<V> IndexMut<RangeFrom<usize>> for EntityIndexMapSlice<V> {
    fn index_mut(&mut self, key: RangeFrom<usize>) -> &mut Self {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { Self::from_slice_unchecked_mut(self.1.index_mut(key)) }
    }
}

impl<V> IndexMut<RangeFull> for EntityIndexMapSlice<V> {
    fn index_mut(&mut self, key: RangeFull) -> &mut Self {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { Self::from_slice_unchecked_mut(self.1.index_mut(key)) }
    }
}

impl<V> IndexMut<RangeInclusive<usize>> for EntityIndexMapSlice<V> {
    fn index_mut(&mut self, key: RangeInclusive<usize>) -> &mut Self {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { Self::from_slice_unchecked_mut(self.1.index_mut(key)) }
    }
}

impl<V> IndexMut<RangeTo<usize>> for EntityIndexMapSlice<V> {
    fn index_mut(&mut self, key: RangeTo<usize>) -> &mut Self {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { Self::from_slice_unchecked_mut(self.1.index_mut(key)) }
    }
}

impl<V> IndexMut<RangeToInclusive<usize>> for EntityIndexMapSlice<V> {
    fn index_mut(&mut self, key: RangeToInclusive<usize>) -> &mut Self {
        // SAFETY: The source IndexMap used EntityHash.
        unsafe { Self::from_slice_unchecked_mut(self.1.index_mut(key)) }
    }
}

impl<V> IndexMut<usize> for EntityIndexMapSlice<V> {
    fn index_mut(&mut self, key: usize) -> &mut V {
        self.1.index_mut(key)
    }
}

pub struct Iter<'a, V, S = EntityHash>(map::Iter<'a, Entity, V>, PhantomData<S>);

impl<V> Iter<'_, V> {
    pub fn as_slice(&self) -> &EntityIndexMapSlice<V> {
        unsafe { EntityIndexMapSlice::from_slice_unchecked(self.0.as_slice()) }
    }
}

impl<'a, V> Iterator for Iter<'a, V> {
    type Item = (&'a Entity, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<V> DoubleEndedIterator for Iter<'_, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl<V> ExactSizeIterator for Iter<'_, V> {}

impl<V> FusedIterator for Iter<'_, V> {}

impl<V> Clone for Iter<'_, V> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<V: Debug> Debug for Iter<'_, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Iter").field(&self.0).field(&self.1).finish()
    }
}

impl<V> Default for Iter<'_, V> {
    fn default() -> Self {
        Self(Default::default(), PhantomData)
    }
}

pub struct IterMut<'a, V, S = EntityHash>(map::IterMut<'a, Entity, V>, PhantomData<S>);

impl<'a, V> IterMut<'a, V> {
    pub fn as_slice(&self) -> &EntityIndexMapSlice<V> {
        unsafe { EntityIndexMapSlice::from_slice_unchecked(self.0.as_slice()) }
    }

    pub fn into_slice(self) -> &'a mut EntityIndexMapSlice<V> {
        unsafe { EntityIndexMapSlice::from_slice_unchecked_mut(self.0.into_slice()) }
    }
}

impl<'a, V> Iterator for IterMut<'a, V> {
    type Item = (&'a Entity, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<V> DoubleEndedIterator for IterMut<'_, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl<V> ExactSizeIterator for IterMut<'_, V> {}

impl<V> FusedIterator for IterMut<'_, V> {}

impl<V: Debug> Debug for IterMut<'_, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("IterMut")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

impl<V> Default for IterMut<'_, V> {
    fn default() -> Self {
        Self(Default::default(), PhantomData)
    }
}

pub struct IntoIter<V, S = EntityHash>(map::IntoIter<Entity, V>, PhantomData<S>);

impl<V> IntoIter<V> {
    pub fn as_slice(&self) -> &EntityIndexMapSlice<V> {
        unsafe { EntityIndexMapSlice::from_slice_unchecked(self.0.as_slice()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut EntityIndexMapSlice<V> {
        unsafe { EntityIndexMapSlice::from_slice_unchecked_mut(self.0.as_mut_slice()) }
    }
}

impl<V> Iterator for IntoIter<V> {
    type Item = (Entity, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<V> DoubleEndedIterator for IntoIter<V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl<V> ExactSizeIterator for IntoIter<V> {}

impl<V> FusedIterator for IntoIter<V> {}

impl<V: Clone> Clone for IntoIter<V> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<V: Debug> Debug for IntoIter<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("IntoIter")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

impl<V> Default for IntoIter<V> {
    fn default() -> Self {
        Self(Default::default(), PhantomData)
    }
}

pub struct Drain<'a, V, S = EntityHash>(map::Drain<'a, Entity, V>, PhantomData<S>);

impl<V> Drain<'_, V> {
    pub fn as_slice(&self) -> &EntityIndexMapSlice<V> {
        unsafe { EntityIndexMapSlice::from_slice_unchecked(self.0.as_slice()) }
    }
}

impl<V> Iterator for Drain<'_, V> {
    type Item = (Entity, V);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<V> DoubleEndedIterator for Drain<'_, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl<V> ExactSizeIterator for Drain<'_, V> {}

impl<V> FusedIterator for Drain<'_, V> {}

impl<V: Debug> Debug for Drain<'_, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Drain")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

pub struct Keys<'a, V, S = EntityHash>(map::Keys<'a, Entity, V>, PhantomData<S>);

impl<'a, V, S> Deref for Keys<'a, V, S> {
    type Target = map::Keys<'a, Entity, V>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, V> Iterator for Keys<'a, V> {
    type Item = &'a Entity;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<V> DoubleEndedIterator for Keys<'_, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl<V> ExactSizeIterator for Keys<'_, V> {}

impl<V> FusedIterator for Keys<'_, V> {}

impl<V> Index<usize> for Keys<'_, V> {
    type Output = Entity;

    fn index(&self, index: usize) -> &Entity {
        self.0.index(index)
    }
}

impl<V> Clone for Keys<'_, V> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

impl<V: Debug> Debug for Keys<'_, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Keys").field(&self.0).field(&self.1).finish()
    }
}

impl<V> Default for Keys<'_, V> {
    fn default() -> Self {
        Self(Default::default(), PhantomData)
    }
}

unsafe impl<V> EntitySet for Keys<'_, V> {}

pub struct IntoKeys<V, S = EntityHash>(map::IntoKeys<Entity, V>, PhantomData<S>);

impl<V, S> Deref for IntoKeys<V, S> {
    type Target = map::IntoKeys<Entity, V>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<V> Iterator for IntoKeys<V> {
    type Item = Entity;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<V> DoubleEndedIterator for IntoKeys<V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back()
    }
}

impl<V> ExactSizeIterator for IntoKeys<V> {}

impl<V> FusedIterator for IntoKeys<V> {}

impl<V: Debug> Debug for IntoKeys<V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("IntoKeys")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

impl<V> Default for IntoKeys<V> {
    fn default() -> Self {
        Self(Default::default(), PhantomData)
    }
}

unsafe impl<V> EntitySet for IntoKeys<V> {}
