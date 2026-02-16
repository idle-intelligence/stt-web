/**
 * Lightweight SentencePiece decoder for browser.
 *
 * Parses a SentencePiece .model file (protobuf format) and decodes token IDs to text.
 * DECODE ONLY — does not support encoding text to tokens.
 *
 * Special tokens:
 *   - id=0: EOS (end of sequence) — skip in decode
 *   - id=3: padding — skip in decode
 *
 * SentencePiece uses ▁ (U+2581, lower one eighth block) to represent spaces.
 */

export class SpmDecoder {
    constructor() {
        this.vocab = null; // Map: token_id -> piece_string
        this.ready = false;
    }

    /**
     * Load and parse a SentencePiece .model file.
     * @param {string} url - URL to the .model file
     */
    async load(url) {
        const resp = await fetch(url);
        if (!resp.ok) {
            throw new Error(`Failed to fetch tokenizer: ${resp.status} ${resp.statusText}`);
        }

        const buf = await resp.arrayBuffer();
        this.vocab = this._parseModelProto(new Uint8Array(buf));
        this.ready = true;
    }

    /**
     * Decode token IDs to text.
     * @param {number[]} ids - Array of token IDs
     * @returns {string} Decoded text
     */
    decode(ids) {
        if (!this.ready) {
            throw new Error('SpmDecoder not loaded. Call load() first.');
        }

        const pieces = [];
        for (const id of ids) {
            // Skip special tokens
            if (id === 0 || id === 3) {
                continue;
            }

            const piece = this.vocab.get(id);
            if (piece !== undefined) {
                pieces.push(piece);
            }
        }

        // Join pieces and convert SentencePiece underscores to spaces
        let text = pieces.join('');
        text = text.replace(/▁/g, ' '); // U+2581 -> space
        return text.trim();
    }

    /**
     * Parse SentencePiece ModelProto from protobuf bytes.
     *
     * Minimal protobuf parser for SentencePiece .model format:
     *   message ModelProto {
     *     repeated SentencePiece pieces = 1;
     *   }
     *   message SentencePiece {
     *     optional string piece = 1;
     *     optional float score = 2;
     *     optional Type type = 3;
     *   }
     *
     * We only need to extract the `piece` strings (field 1) and map them to IDs.
     * Token ID = index in the pieces array.
     *
     * @param {Uint8Array} buf - Protobuf bytes
     * @returns {Map<number, string>} Map of token_id -> piece_string
     */
    _parseModelProto(buf) {
        const vocab = new Map();
        let pos = 0;
        let tokenId = 0;

        while (pos < buf.length) {
            // Read varint-encoded tag
            const { value: tag, length: tagLen } = this._readVarint(buf, pos);
            pos += tagLen;

            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (fieldNum === 1 && wireType === 2) {
                // Field 1, wire type 2 (length-delimited) = repeated SentencePiece
                const { value: len, length: lenLen } = this._readVarint(buf, pos);
                pos += lenLen;

                const pieceBytes = buf.slice(pos, pos + len);
                const piece = this._parseSentencePiece(pieceBytes);
                vocab.set(tokenId, piece);
                tokenId++;

                pos += len;
            } else {
                // Skip unknown fields
                pos = this._skipField(buf, pos, wireType);
            }
        }

        return vocab;
    }

    /**
     * Parse a single SentencePiece message.
     * We only extract field 1 (piece string).
     */
    _parseSentencePiece(buf) {
        let pos = 0;
        let piece = '';

        while (pos < buf.length) {
            const { value: tag, length: tagLen } = this._readVarint(buf, pos);
            pos += tagLen;

            const fieldNum = tag >> 3;
            const wireType = tag & 0x7;

            if (fieldNum === 1 && wireType === 2) {
                // Field 1, wire type 2 (length-delimited) = piece string
                const { value: len, length: lenLen } = this._readVarint(buf, pos);
                pos += lenLen;

                const strBytes = buf.slice(pos, pos + len);
                piece = new TextDecoder('utf-8').decode(strBytes);
                pos += len;
            } else {
                pos = this._skipField(buf, pos, wireType);
            }
        }

        return piece;
    }

    /**
     * Read a varint-encoded unsigned integer.
     * Returns { value, length }.
     */
    _readVarint(buf, pos) {
        let value = 0;
        let shift = 0;
        let len = 0;

        while (pos + len < buf.length) {
            const byte = buf[pos + len];
            len++;

            value |= (byte & 0x7f) << shift;
            shift += 7;

            if ((byte & 0x80) === 0) {
                break;
            }
        }

        return { value, length: len };
    }

    /**
     * Skip a field based on wire type.
     * Returns new position.
     */
    _skipField(buf, pos, wireType) {
        switch (wireType) {
            case 0: // Varint
                while (pos < buf.length && (buf[pos] & 0x80) !== 0) {
                    pos++;
                }
                return pos + 1;

            case 1: // 64-bit
                return pos + 8;

            case 2: // Length-delimited
                const { value: len, length: lenLen } = this._readVarint(buf, pos);
                return pos + lenLen + len;

            case 5: // 32-bit
                return pos + 4;

            default:
                throw new Error(`Unknown wire type: ${wireType}`);
        }
    }
}
