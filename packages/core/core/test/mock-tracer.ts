import {
  AttributeValue,
  Attributes,
  Context,
  Span,
  SpanContext,
  SpanOptions,
  Tracer,
} from '@opentelemetry/api';

export class MockTracer implements Tracer {
  spans: MockSpan[] = [];

  get jsonSpans() {
    return this.spans.map(span => ({
      name: span.name,
      attributes: span.attributes,
    }));
  }

  startSpan(name: string, options?: SpanOptions, context?: Context): Span {
    const span = new MockSpan({
      name,
      options,
      context,
    });
    this.spans.push(span);
    return span;
  }

  startActiveSpan<F extends (span: Span) => unknown>(
    name: string,
    arg1: unknown,
    arg2?: unknown,
    arg3?: F,
  ): ReturnType<any> {
    if (typeof arg1 === 'function') {
      const span = new MockSpan({
        name,
      });
      this.spans.push(span);
      return arg1(span);
    }
    if (typeof arg2 === 'function') {
      const span = new MockSpan({
        name,
        options: arg1 as SpanOptions,
      });
      this.spans.push(span);
      return arg2(span);
    }
    if (typeof arg3 === 'function') {
      const span = new MockSpan({
        name,
        options: arg1 as SpanOptions,
        context: arg2 as Context,
      });
      this.spans.push(span);
      return arg3(span);
    }
  }
}

class MockSpan implements Span {
  name: string;
  context?: Context;
  options?: SpanOptions;
  attributes: Attributes = {};

  readonly _spanContext: SpanContext = new MockSpanContext();

  constructor({
    name,
    options,
    context,
  }: {
    name: string;
    options?: SpanOptions;
    context?: Context;
  }) {
    this.name = name;
    this.context = context;
    this.options = options;
  }

  spanContext(): SpanContext {
    return this._spanContext;
  }

  setAttribute(key: string, value: AttributeValue): this {
    this.attributes = { ...this.attributes, [key]: value };
    return this;
  }

  setAttributes(attributes: Attributes): this {
    this.attributes = { ...this.attributes, ...attributes };
    return this;
  }

  addEvent() {
    return this;
  }
  addLink() {
    return this;
  }
  addLinks() {
    return this;
  }
  setStatus() {
    return this;
  }
  updateName() {
    return this;
  }
  end() {
    return this;
  }
  isRecording() {
    return false;
  }
  recordException() {
    return this;
  }
}

class MockSpanContext implements SpanContext {
  traceId = 'test-trace-id';
  spanId = 'test-span-id';
  traceFlags = 0;
}
