// @ts-nocheck
import { generateText, streamText } from 'ai';

// Test generateText with both roundtrip types
await generateText({
  model,
  maxSteps: 4
});

// Test streamText with just maxToolRoundtrips
await streamText({
  model,
  maxSteps: 6
});

// Test generateText with just maxAutomaticRoundtrips
await generateText({
  model,
  maxSteps: 5
});

// Test property access
const result = await generateText({ model });
console.log(result.steps.length);
