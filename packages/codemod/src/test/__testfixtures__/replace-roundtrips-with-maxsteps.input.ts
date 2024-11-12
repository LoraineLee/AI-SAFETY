// @ts-nocheck
import { generateText, streamText } from 'ai';

// Test generateText with both roundtrip types
await generateText({
  model,
  maxToolRoundtrips: 3,
  maxAutomaticRoundtrips: 2
});

// Test streamText with just maxToolRoundtrips
await streamText({
  model,
  maxToolRoundtrips: 5
});

// Test generateText with just maxAutomaticRoundtrips
await generateText({
  model,
  maxAutomaticRoundtrips: 4
});

// Test property access
const result = await generateText({ model });
console.log(result.roundtrips.length);
