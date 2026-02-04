mod util;

use std::{
    fmt::Debug,
    ops::{Index, IndexMut},
};

use crate::util::{flat_mut, flat_ref};

pub struct RingBuffer<T> {
    buffer: Vec<Option<T>>,
    capacity: usize,
    head: usize,
    count: usize,
}
impl<T> RingBuffer<T> {
    fn offset(&self, offset: usize) -> usize {
        let offset = offset.min(self.count);
        (self.head + offset) % self.capacity
    }

    pub fn new(capacity: usize) -> Self {
        let mut buffer = Vec::with_capacity(capacity);
        buffer.resize_with(capacity, || None);
        RingBuffer {
            buffer,
            capacity,
            head: 0,
            count: 0,
        }
    }

    pub fn from_vec_raw(vec: Vec<T>) -> Self {
        let capacity = vec.len();
        Self {
            buffer: vec.into_iter().map(Some).collect(),
            capacity,
            head: 0,
            count: capacity,
        }
    }

    pub fn to_vec(mut self) -> Vec<T> {
        let mut vec = Vec::with_capacity(self.count);
        for offset in 0..self.count {
            let ix = self.offset(offset);
            let Some(item) = self.buffer[ix].take() else {
                break;
            };
            vec.push(item);
        }

        assert_eq!(
            vec.len(),
            self.count,
            "The length of the vector should match the count"
        );

        vec
    }

    pub fn to_vec_ref(&self) -> Vec<&T> {
        let mut vec = Vec::with_capacity(self.count);
        for offset in 0..self.count {
            let ix = self.offset(offset);
            let Some(item) = self.buffer[ix].as_ref() else {
                break;
            };
            vec.push(item);
        }

        assert_eq!(
            vec.len(),
            self.count,
            "The length of the vector should match the count"
        );

        vec
    }

    pub fn push(&mut self, value: T) {
        let tail = self.offset(self.count);
        if self.count == self.capacity {
            self.head = (self.head + 1) % self.capacity;
        } else {
            self.count += 1;
        }
        self.buffer[tail] = Some(value);
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.count == 0 {
            self.head = self.head.saturating_sub(1);
        } else {
            self.count = self.count.saturating_sub(1);
        }
        let tail = self.offset(self.count);
        self.buffer[tail].take()
    }

    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.fill_from(std::iter::repeat_n(value, self.capacity));
    }

    pub fn fill_with(&mut self, mut f: impl FnMut(usize) -> T) {
        let iter = (0..).map(|o| f(o)).take(self.capacity);
        self.fill_from(iter);
    }

    pub fn fill_from<I>(&mut self, iter: I)
    where
        I: Iterator<Item = T>,
    {
        for item in iter {
            self.push(item);
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn clear(&mut self) {
        self.count = 0;
        self.head = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn is_full(&self) -> bool {
        self.count == self.capacity
    }

    pub fn replace(&mut self, index: usize, value: T) -> Option<T> {
        if index > self.count {
            None
        } else {
            self.replace_unchecked(index, value)
        }
    }

    pub fn replace_unchecked(&mut self, index: usize, value: T) -> Option<T> {
        if index > self.count {
            panic!("Index out of bounds");
        }

        let offset = self.offset(index);
        let v = (&mut self.buffer)[offset].take();

        (&mut self.buffer)[offset] = Some(value);

        v
    }

    pub fn first(&self) -> Option<&T> {
        self.get(0)
    }

    pub fn first_mut(&mut self) -> Option<&mut T> {
        self.get_mut(0)
    }

    pub fn last(&self) -> Option<&T> {
        self.get(self.count - 1)
    }

    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.get_mut(self.count - 1)
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len() {
            None
        } else {
            let offset = self.offset(index);
            flat_ref(self.buffer.get(offset))
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.len() {
            None
        } else {
            let offset = self.offset(index);
            flat_mut(self.buffer.get_mut(offset))
        }
    }

    pub unsafe fn into_raw_buffer(&mut self) -> &mut Vec<Option<T>> {
        &mut self.buffer
    }

    pub unsafe fn set_count(&mut self, count: usize) {
        self.count = count;
    }

    pub fn iter<'a>(&'a self) -> RingBufferIter<'a, T> {
        RingBufferIter {
            buffer: self,
            offset: 0,
        }
    }
}

impl<T> Index<usize> for RingBuffer<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("index out of bounds")
    }
}

impl<T> IndexMut<usize> for RingBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut <Self as Index<usize>>::Output {
        self.get_mut(index).expect("index out of bounds")
    }
}

impl<T: PartialEq> PartialEq for RingBuffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T: Clone> Clone for RingBuffer<T> {
    fn clone(&self) -> Self {
        let mut buffer = RingBuffer::new(self.capacity);
        buffer.fill_from(self.into_iter().cloned());
        buffer
    }
}

impl<T> From<Vec<T>> for RingBuffer<T> {
    fn from(vec: Vec<T>) -> Self {
        Self::from_vec_raw(vec)
    }
}

impl<'a, T> From<&'a [T]> for RingBuffer<&'a T> {
    fn from(slice: &'a [T]) -> Self {
        let mut buffer = RingBuffer::new(slice.len());
        buffer.fill_from(slice.iter());
        buffer
    }
}

#[cfg(feature = "serde")]
mod _serde_impl {
    use serde::{Deserialize, Serialize};

    use crate::RingBuffer;
    impl<T: Serialize> Serialize for RingBuffer<T> {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            serializer.collect_seq(self)
        }
    }

    impl<'de, T> Deserialize<'de> for RingBuffer<T>
    where
        T: Deserialize<'de>,
    {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let vec = Vec::deserialize(deserializer)?;
            Ok(Self::from_vec_raw(vec))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_serde() {
            let buffer: RingBuffer<i32> = RingBuffer::from(vec![1, 2, 3]);
            let serialized = serde_json::to_string(&buffer).unwrap();
            let deserialized: RingBuffer<i32> = serde_json::from_str(&serialized).unwrap();
            assert_eq!(buffer, deserialized);
        }
    }
}

impl<T: Debug> Debug for RingBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        let mut iter = self.into_iter().peekable();

        while let Some(item) = iter.next() {
            write!(f, "{:?}", item)?;
            if iter.peek().is_some() {
                write!(f, ", ")?;
            }
        }

        write!(f, "]")?;

        Ok(())
    }
}

impl<'a, T> IntoIterator for &'a RingBuffer<T> {
    type Item = &'a T;
    type IntoIter = RingBufferIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        RingBufferIter {
            buffer: self,
            offset: 0,
        }
    }
}

pub struct RingBufferIter<'a, T> {
    buffer: &'a RingBuffer<T>,
    offset: usize,
}

impl<'a, T> RingBufferIter<'a, T> {
    pub fn new(buffer: &'a RingBuffer<T>) -> Self {
        RingBufferIter { buffer, offset: 0 }
    }
}

impl<'a, T> Iterator for RingBufferIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.buffer.count {
            return None;
        }

        let ix = self.buffer.offset(self.offset);
        self.offset += 1;
        self.buffer.buffer.get(ix)?.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_pop() {
        let mut buffer = RingBuffer::new(3);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        assert_eq!(buffer.pop(), Some(3));
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.pop(), Some(1));
        assert_eq!(buffer.pop(), None);
    }

    #[test]
    fn test_iter() {
        let mut buffer = RingBuffer::new(3);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        let mut iter = buffer.into_iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_wrap() {
        let mut buffer = RingBuffer::new(2);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        buffer.push(4);

        let mut iter = buffer.into_iter();
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_to_vec() {
        let mut buffer = RingBuffer::new(3);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        let vec = buffer.to_vec();
        assert_eq!(vec, vec![1, 2, 3]);

        let mut buffer = RingBuffer::new(3);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        unsafe {
            buffer.set_count(2);
        }
        let vec = buffer.to_vec();
        assert_eq!(vec, vec![1, 2]);

        let mut buffer = RingBuffer::new(3);
        buffer.push(1);
        buffer.push(2);

        let vec = buffer.to_vec();
        assert_eq!(vec, vec![1, 2]);
    }

    #[test]
    fn test_to_vec_ref() {
        let mut buffer = RingBuffer::new(3);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        let vec = buffer.to_vec_ref();
        assert_eq!(vec, vec![&1, &2, &3]);
    }

    #[test]
    fn test_debug_defer() {
        let mut buffer = RingBuffer::new(3);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);

        let debug = format!("{:?}", buffer);
        let expected = format!("{:?}", vec![1, 2, 3]);
        assert_eq!(debug, expected);
    }

    #[test]
    fn test_clear() {
        let mut buffer = RingBuffer::new(3);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        buffer.clear();
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_is_empty() {
        let mut buffer = RingBuffer::new(3);
        assert!(buffer.is_empty());
        buffer.push(1);
        assert!(!buffer.is_empty());
        buffer.clear();
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_is_full() {
        let mut buffer = RingBuffer::new(3);
        assert!(!buffer.is_full());
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        assert!(buffer.is_full());
    }

    #[test]
    fn test_replace() {
        let mut buffer = RingBuffer::new(3);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        let result = buffer.replace(0, 4);
        assert_eq!(buffer.to_vec_ref(), vec![&4, &2, &3]);
        assert_eq!(result, Some(1));
        let result = buffer.replace(5, 4);
        assert_eq!(buffer.to_vec_ref(), vec![&4, &2, &3]);
        assert_eq!(result, None);
    }

    #[test]
    fn test_replace_unchecked() {
        let mut buffer = RingBuffer::new(3);
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        let result = buffer.replace_unchecked(0, 4);
        assert_eq!(buffer.to_vec_ref(), vec![&4, &2, &3]);
        assert_eq!(result, Some(1));
    }

    #[test]
    fn test_first() {
        let mut buffer = RingBuffer::new(3);
        buffer.push(1); // [<1>,  _ ,  _ ]
        assert_eq!(buffer.first(), Some(&1));
        buffer.push(2); // [<1>,  2 ,  _ ]
        assert_eq!(buffer.first(), Some(&1));
        buffer.push(3); // [<1>,  2 ,  3 ]
        assert_eq!(buffer.first(), Some(&1));
        buffer.push(2); // [ 2 , <2>,  3 ]
        assert_eq!(buffer.first(), Some(&2));
        buffer.push(3); // [ 2 ,  3 , <3>]
        assert_eq!(buffer.first(), Some(&3));
        buffer.push(3); // [<2>,  3,   3 ]
        assert_eq!(buffer.first(), Some(&2));
    }

    #[test]
    fn test_last() {
        let mut buffer = RingBuffer::new(3);
        buffer.push(1); // [<1>,  _ ,  _ ]
        assert_eq!(buffer.last(), Some(&1));
        buffer.push(2); // [ 1 , <2>,  _ ]
        assert_eq!(buffer.last(), Some(&2));
        buffer.push(3); // [ 1 ,  2 , <3>]
        assert_eq!(buffer.last(), Some(&3));
        buffer.push(2); // [<2>,  2 ,  3 ]
        assert_eq!(buffer.last(), Some(&2));
        buffer.push(3); // [ 2 ,  <3>,  3 ]
        assert_eq!(buffer.last(), Some(&3));
        buffer.push(3); // [ 2 ,  3,  <3>]
        assert_eq!(buffer.last(), Some(&3));
    }

    #[test]
    fn test_get() {
        let mut buffer = RingBuffer::new(3);
        buffer.push(1);
        assert_eq!(buffer.get(0), Some(&1));
        assert_eq!(buffer.get(1), None);
        assert_eq!(buffer.get(2), None);

        assert_eq!(buffer.first_mut(), Some(&mut 1));
        assert_eq!(buffer.last_mut(), Some(&mut 1));

        assert_eq!(buffer.get_mut(0), Some(&mut 1));
        assert_eq!(buffer.get_mut(1), None);
        assert_eq!(buffer.get_mut(2), None);
    }

    #[test]
    fn test_fill() {
        let mut buffer = RingBuffer::new(3);
        buffer.fill(1);
        assert_eq!(buffer.get(0), Some(&1));
        assert_eq!(buffer.get(1), Some(&1));
        assert_eq!(buffer.get(2), Some(&1));
    }

    #[test]
    fn test_fill_from() {
        let mut buffer = RingBuffer::new(3);
        buffer.fill_from(std::iter::repeat(2).take(3));
        assert_eq!(buffer.get(0), Some(&2));
        assert_eq!(buffer.get(1), Some(&2));
        assert_eq!(buffer.get(2), Some(&2));
    }

    #[test]
    fn test_fill_from_partial_fill() {
        let mut buffer = RingBuffer::new(30);
        buffer.fill_from(std::iter::repeat_n(3, 3));
        assert_eq!(buffer.get(0), Some(&3));
        assert_eq!(buffer.get(1), Some(&3));
        assert_eq!(buffer.get(2), Some(&3));
        assert_eq!(buffer.get(3), None);
    }

    #[test]
    fn test_fill_from_overfill() {
        let mut buffer = RingBuffer::new(3);
        buffer.fill_from(0..5);
        assert_eq!(buffer.get(0), Some(&2));
        assert_eq!(buffer.get(1), Some(&3));
        assert_eq!(buffer.get(2), Some(&4));
    }

    #[test]
    fn test_fill_with() {
        let mut buffer = RingBuffer::new(3);
        buffer.fill_with(|i| i * 2);
        assert_eq!(buffer.get(0), Some(&0));
        assert_eq!(buffer.get(1), Some(&2));
        assert_eq!(buffer.get(2), Some(&4));
    }

    #[test]
    fn test_from_vec_raw() {
        let buffer = RingBuffer::from_vec_raw(vec![1, 2, 3]);
        assert_eq!(buffer.get(0), Some(&1));
        assert_eq!(buffer.get(1), Some(&2));
        assert_eq!(buffer.get(2), Some(&3));
    }
}

#[cfg(test)]
mod trait_tests {
    use super::*;

    #[test]
    fn test_from_slice() {
        let data = vec![1, 2, 3, 4];

        let buffer = RingBuffer::from(&data[2..]);
        assert_eq!(buffer.get(0), Some(&&3));
        assert_eq!(buffer.get(1), Some(&&4));
    }

    #[test]
    fn test_from_vec() {
        let buffer = RingBuffer::from(vec![1, 2, 3]);
        assert_eq!(buffer.get(0), Some(&1));
        assert_eq!(buffer.get(1), Some(&2));
        assert_eq!(buffer.get(2), Some(&3));
    }

    #[test]
    fn test_clone() {
        let buffer = RingBuffer::from(vec![1, 2, 3]);
        let cloned_buffer = buffer.clone();
        assert_eq!(cloned_buffer.get(0), Some(&1));
        assert_eq!(cloned_buffer.get(1), Some(&2));
        assert_eq!(cloned_buffer.get(2), Some(&3));
    }

    #[test]
    fn test_eq() {
        let buffer1 = RingBuffer::from(vec![1, 2, 3]);
        let buffer2 = RingBuffer::from(vec![1, 2, 3]);
        assert_eq!(buffer1, buffer2);
    }

    #[test]
    fn test_indices() {
        let mut buffer = RingBuffer::from(vec![1, 2, 3]);
        assert_eq!(&buffer[0], &1);
        assert_eq!(&mut buffer[1], &mut 2);
    }
}

#[cfg(test)]
mod unsafe_tests {

    use super::*;

    #[test]
    fn test_into_buffer_raw() {
        let mut buffer = RingBuffer::new(3);
        buffer.fill_from(0..4);
        assert_eq!(buffer.get(0), Some(&1));
        assert_eq!(buffer.get(1), Some(&2));
        assert_eq!(buffer.get(2), Some(&3));

        let raw_buffer = unsafe { buffer.into_raw_buffer() };
        assert_eq!(raw_buffer[0], Some(3));
        assert_eq!(raw_buffer[1], Some(1));
        assert_eq!(raw_buffer[2], Some(2));
    }

    #[test]
    fn test_set_count() {
        let mut buffer = RingBuffer::new(3);
        buffer.fill_from(0..4);
        assert_eq!(buffer.get(0), Some(&1));
        assert_eq!(buffer.get(1), Some(&2));
        assert_eq!(buffer.get(2), Some(&3));

        unsafe {
            buffer.set_count(2);
        }
        assert_eq!(buffer.get(0), Some(&1));
        assert_eq!(buffer.get(1), Some(&2));
        assert_eq!(buffer.get(2), None);
    }
}
