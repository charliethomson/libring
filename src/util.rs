/// Some(&Some(42)) => Some(&42)
pub fn flat_ref<T>(this: Option<&Option<T>>) -> Option<&T> {
    match this {
        Some(inner) => inner.as_ref(),
        None => None,
    }
}

/// Some(&mut Some(42)) => Some(&mut 42)
pub fn flat_mut<T>(this: Option<&mut Option<T>>) -> Option<&mut T> {
    match this {
        Some(inner) => inner.as_mut(),
        None => None,
    }
}

/// Some(&mut Some(42)) => Some(42)
pub fn flat_take<T>(this: Option<&mut Option<T>>) -> Option<T> {
    match this {
        Some(inner) => inner.take(),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_ref() {
        let option = Some(&Some(42));
        let flat_ref = flat_ref(option);
        assert_eq!(flat_ref, Some(&42));
    }

    #[test]
    fn test_flat_ref_negatives() {
        let option: Option<&Option<i32>> = Some(&None);
        assert_eq!(flat_ref(option), None);
        let option: Option<&Option<i32>> = None;
        assert_eq!(flat_ref(option), None);
    }

    #[test]
    fn test_flat_mut() {
        let option = Some(&mut Some(42));
        let flat_mut = flat_mut(option);
        assert_eq!(flat_mut, Some(&mut 42));
    }

    #[test]
    fn test_flat_mut_negatives() {
        let option: Option<&mut Option<i32>> = Some(&mut None);
        assert_eq!(flat_mut(option), None);
        let option: Option<&mut Option<i32>> = None;
        assert_eq!(flat_mut(option), None);
    }

    #[test]
    fn test_flat_take() {
        let option = Some(&mut Some(42));
        let flat_take = flat_take(option);
        assert_eq!(flat_take, Some(42));
    }

    #[test]
    fn test_flat_take_negatives() {
        let option: Option<&mut Option<i32>> = Some(&mut None);
        assert_eq!(flat_take(option), None);
        let option: Option<&mut Option<i32>> = None;
        assert_eq!(flat_take(option), None);
    }
}
