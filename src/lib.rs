use std::fmt::Debug;

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
        self.fill_from(std::iter::repeat(value));
    }

    pub fn fill_with(&mut self, mut f: impl FnMut() -> T) {
        self.fill_from(std::iter::from_fn(|| Some(f())).take(self.capacity));
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

    pub fn iter<'a>(&'a self) -> RingBufferIter<'a, T> {
        RingBufferIter {
            buffer: self,
            offset: 0,
        }
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
}
