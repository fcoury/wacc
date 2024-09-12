pub fn safe_split_at<T>(slice: &[T], mid: usize) -> (&[T], &[T]) {
    if mid > slice.len() {
        (slice, &[])
    } else {
        slice.split_at(mid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_split_at() {
        let slice = [1, 2, 3, 4, 5];
        assert_eq!(safe_split_at(&slice, 0), (&slice[0..0], &slice[0..5]));
        assert_eq!(safe_split_at(&slice, 1), (&slice[0..1], &slice[1..5]));
        assert_eq!(safe_split_at(&slice, 5), (&slice[0..5], &slice[0..0]));
        assert_eq!(safe_split_at(&slice, 6), (&slice[0..5], &slice[0..0]));
    }
}
