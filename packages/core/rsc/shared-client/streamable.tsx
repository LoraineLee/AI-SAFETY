import { STREAMABLE_VALUE_TYPE } from '../constants';
import type { StreamableValue } from '../types';

function assertStreamableValue(
  value: unknown,
): asserts value is StreamableValue {
  if (
    !value ||
    typeof value !== 'object' ||
    !('type' in value) ||
    value.type !== STREAMABLE_VALUE_TYPE
  ) {
    throw new Error(
      'Invalid value: this hook only accepts values created via `createStreamableValue` from the server.',
    );
  }
}

/**
 * `readStreamableValue` takes a streamable value created via the `createStreamableValue().value` API,
 * and returns an async iterator.
 *
 * ```js
 * // Inside your AI action:
 *
 * async function action() {
 *   'use server'
 *   const streamable = createStreamableValue();
 *
 *   streamable.update(1);
 *   streamable.update(2);
 *   streamable.done(3);
 *   // ...
 *   return streamable.value;
 * }
 * ```
 *
 * And to read the value:
 *
 * ```js
 * const streamableValue = await action()
 * for await (const v of readStreamableValue(streamableValue)) {
 *   console.log(v)
 * }
 * ```
 *
 * This logs out 1, 2, 3 on console.
 */
export function readStreamableValue<T = unknown>(
  streamableValue: StreamableValue<T>,
): AsyncIterable<T | undefined> {
  assertStreamableValue(streamableValue);

  return {
    [Symbol.asyncIterator]() {
      let row: StreamableValue<T> | Promise<StreamableValue<T>> =
        streamableValue;
      let curr = row.curr;
      let done = false;
      let initial = true;

      return {
        async next() {
          if (done) return { value: curr, done: true };

          row = await row;

          if (typeof row.error !== 'undefined') {
            throw row.error;
          }
          if ('curr' in row || row.diff) {
            if (row.diff) {
              switch (row.diff[0]) {
                case 0:
                  if (typeof curr !== 'string') {
                    throw new Error(
                      'Invalid patch: can only append to string types. This is a bug in the AI SDK.',
                    );
                  } else {
                    (curr as string) = curr + row.diff[1];
                  }
                  break;
              }
            } else {
              curr = row.curr;
            }

            // The last emitted { done: true } won't be used as the value
            // by the for await...of syntax.
            if (!row.next) {
              done = true;
              return {
                value: curr,
                done: false,
              };
            }
          }

          if (!row.next) {
            return {
              value: curr,
              done: true,
            };
          }

          row = row.next;
          if (initial) {
            initial = false;
            if (typeof curr === 'undefined') {
              // This is the initial chunk and there isn't an initial value yet.
              // Let's skip this one.
              return this.next();
            }
          }

          return {
            value: curr,
            done: false,
          };
        },
      };
    },
  };
}
