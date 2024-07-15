import { Attachment } from '@ai-sdk/ui-utils';
import { ImagePart, TextPart } from './content-part';

type ContentPart = TextPart | ImagePart;

/**
 * Converts a data URL of type text/* to a text string.
 */
export function dataUrlToText(dataUrl: string): string {
  return Buffer.from(dataUrl, 'base64').toString('utf-8');
}

/**
 * Checks if a string is a Data URL.
 */
function isDataURL(url: string): boolean {
  return new URL(url).protocol === 'data:';
}

/**
 * Converts a list of attachments to a list of content parts
 * for consumption by ai/core functions.
 * Currently only supports images and text attachments.
 */
export function attachmentsToParts(attachments: Attachment[]): ContentPart[] {
  const parts: ContentPart[] = [];

  for (const attachment of attachments) {
    if (attachment.mimeType?.startsWith('image/')) {
      parts.push({ type: 'image', image: attachment.url });
    }

    if (attachment.mimeType?.startsWith('text/') && isDataURL(attachment.url)) {
      parts.push({ type: 'text', text: dataUrlToText(attachment.url) });
    }
  }

  return parts;
}
