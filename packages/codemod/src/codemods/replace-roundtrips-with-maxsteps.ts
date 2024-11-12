import { API, FileInfo } from 'jscodeshift';

export default function transformer(file: FileInfo, api: API) {
  const j = api.jscodeshift;
  const root = j(file.source);

  root
    .find(j.CallExpression)
    .filter(
      path =>
        path.node.callee.type === 'Identifier' &&
        ['generateText', 'streamText'].includes(path.node.callee.name),
    )
    .forEach(path => {
      const optionsArg = path.node.arguments[0];
      if (optionsArg?.type !== 'ObjectExpression') return;

      // Extract roundtrip values before removing
      let maxStepsValue = 1;
      let foundRoundtrips = false;

      optionsArg.properties = optionsArg.properties.filter(prop => {
        if (
          prop.type === 'ObjectProperty' &&
          prop.key.type === 'Identifier' &&
          ['maxToolRoundtrips', 'maxAutomaticRoundtrips'].includes(
            prop.key.name,
          )
        ) {
          foundRoundtrips = true;
          if (prop.value.type === 'NumericLiteral') {
            // maxToolRoundtrips takes precedence over maxAutomaticRoundtrips
            if (prop.key.name === 'maxToolRoundtrips' || maxStepsValue === 1) {
              maxStepsValue = prop.value.value + 1; // Add 1 to roundtrips value
            }
          }
          return false; // Remove the property
        }
        return true;
      });

      // Only add maxSteps if we found and removed roundtrip properties
      if (foundRoundtrips) {
        optionsArg.properties.push(
          j.objectProperty(
            j.identifier('maxSteps'),
            j.numericLiteral(maxStepsValue),
          ),
        );
      }
    });

  // Replace property access
  root
    .find(j.MemberExpression)
    .filter(
      path =>
        path.node.property.type === 'Identifier' &&
        path.node.property.name === 'roundtrips',
    )
    .forEach(path => {
      if (path.node.property.type === 'Identifier') {
        path.node.property.name = 'steps';
      }
    });

  return root.toSource();
}
