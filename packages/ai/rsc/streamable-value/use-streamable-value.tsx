import { startTransition, useLayoutEffect, useState } from 'react';
import { readStreamableValue } from './read-streamable-value';
import { isStreamableValue, StreamableValue } from './streamable-value';

/**
 * `useStreamableValue` is a React hook that takes a streamable value created via the `createStreamableValue().value` API,
 * and returns the current value, error, and pending state.
 *
 * This is useful for consuming streamable values received from a component's props. For example:
 *
 * ```js
 * function MyComponent({ streamableValue }) {
 *   const [data, error, pending] = useStreamableValue(streamableValue);
 *
 *   if (pending) return <div>Loading...</div>;
 *   if (error) return <div>Error: {error.message}</div>;
 *
 *   return <div>Data: {data}</div>;
 * }
 * ```
 */
export function useStreamableValue<T = unknown, Error = unknown>(
  streamableValue?: StreamableValue<T>,
): [data: T | undefined, error: Error | undefined, pending: boolean] {
  const [curr, setCurr] = useState<T | undefined>(
    isStreamableValue(streamableValue) ? streamableValue.curr : undefined,
  );
  const [error, setError] = useState<Error | undefined>(
    isStreamableValue(streamableValue) ? streamableValue.error : undefined,
  );
  const [pending, setPending] = useState<boolean>(
    isStreamableValue(streamableValue) ? !!streamableValue.next : false,
  );

  useLayoutEffect(() => {
    if (!isStreamableValue(streamableValue)) return;

    let cancelled = false;

    const iterator = readStreamableValue(streamableValue);
    if (streamableValue.next) {
      startTransition(() => {
        if (cancelled) return;
        setPending(true);
      });
    }

    (async () => {
      try {
        for await (const value of iterator) {
          if (cancelled) return;
          startTransition(() => {
            if (cancelled) return;
            setCurr(value);
          });
        }
      } catch (e) {
        if (cancelled) return;
        startTransition(() => {
          if (cancelled) return;
          setError(e as Error);
        });
      } finally {
        if (cancelled) return;
        startTransition(() => {
          if (cancelled) return;
          setPending(false);
        });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [streamableValue]);

  return [curr, error, pending];
}
