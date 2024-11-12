import { API, FileInfo } from 'jscodeshift';
import { removeFacade } from './lib/remove-facade';

export default function transformer(fileInfo: FileInfo, api: API) {
  return removeFacade(fileInfo, api, {
    packageName: 'anthropic',
    className: 'Anthropic',
    createFnName: 'createAnthropic',
    methodNames: ['messages', 'chat'],
  });
}
