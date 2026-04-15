#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_INPUT       1024
#define MAX_SEGMENTS    64
#define MAX_SEG_LEN     256
#define MAX_WORDS       256
#define MAX_WORD_LEN    64
#define OUTPUT_FILE     "data_preprocessing.json"

typedef struct {
    const char *from;
    const char *to;
} ReplaceRule;

/* ---------------- NORMALIZER RULES ---------------- */
static ReplaceRule RULES[] = {
    {"go to", "move to"},
    {"walk to", "move to"},
    {"run to", "move to"},
    {"pick up", "pickup"},
    {"grab", "pickup"},
    {"talk to", "interact with"},
    {"speak to", "interact with"},
    {"hit", "attack"},
    {"strike", "attack"},
    {"fight", "attack"},
    {"eliminate", "attack"},
    {"first", "1st"},
    {"second", "2nd"},
    {NULL, NULL}
};

/* ---------------- SEGMENTATION TABLES ---------------- */
static const char *STRONG_SEPARATORS[] = {
    " then ", " but ", " also ", " next ", " after that ",
    NULL
};

static const char *ACTIONS[] = {
    "attack", "heal", "move", "follow", "interact", "defend", "talk", "use",
    NULL
};

static const char *TARGETS[] = {
    "enemy", "wizard", "soldier", "guard", "orc", "goblin", "target", "mage",
    NULL
};

static const char *REFERENCE_WORDS[] = {
    "one", "ones", "him", "her", "them",
    NULL
};

static const char *COLORS[] = {
    "red", "blue", "green", "yellow", "black", "white", "purple", "orange",
    NULL
};

/* ---------------- BASIC HELPERS ---------------- */
void trim_whitespace(char *s) {
    int start = 0;
    while (s[start] && isspace((unsigned char)s[start])) start++;

    if (start > 0) {
        memmove(s, s + start, strlen(s + start) + 1);
    }

    int end = (int)strlen(s) - 1;
    while (end >= 0 && isspace((unsigned char)s[end])) {
        s[end] = '\0';
        end--;
    }
}

void lowercaser(char *str) {
    while (*str) {
        *str = (char)tolower((unsigned char)*str);
        str++;
    }
}

void to_lower_copy(const char *src, char *dst, int max_len) {
    int i;
    for (i = 0; i < max_len - 1 && src[i]; i++) {
        dst[i] = (char)tolower((unsigned char)src[i]);
    }
    dst[i] = '\0';
}

int is_word_char(char c) {
    return isalnum((unsigned char)c) || c == '_';
}

int starts_with_word(const char *text, const char *word) {
    int len = (int)strlen(word);
    if (strncmp(text, word, len) != 0) return 0;
    return text[len] == '\0' || !is_word_char(text[len]);
}

int contains_word(const char *text, const char *word) {
    int wlen = (int)strlen(word);
    const char *p = text;

    while ((p = strstr(p, word)) != NULL) {
        int left_ok  = (p == text) || !is_word_char(*(p - 1));
        int right_ok = !is_word_char(*(p + wlen));
        if (left_ok && right_ok) return 1;
        p++;
    }
    return 0;
}

int starts_with_any(const char *text, const char *list[]) {
    while (*text && isspace((unsigned char)*text)) text++;

    for (int i = 0; list[i] != NULL; i++) {
        if (starts_with_word(text, list[i])) return 1;
    }
    return 0;
}

int has_action(const char *text) {
    return starts_with_any(text, ACTIONS) || contains_word(text, "interact");
}

int has_reference_word(const char *text) {
    for (int i = 0; REFERENCE_WORDS[i] != NULL; i++) {
        if (contains_word(text, REFERENCE_WORDS[i])) return 1;
    }
    return 0;
}

int starts_like_new_command_or_target(const char *text) {
    while (*text && isspace((unsigned char)*text)) text++;

    if (starts_with_any(text, ACTIONS)) return 1;
    if (starts_with_word(text, "the")) return 1;
    if (starts_with_word(text, "a")) return 1;
    if (starts_with_word(text, "an")) return 1;
    if (starts_with_any(text, REFERENCE_WORDS)) return 1;

    return 0;
}

int starts_with_color(const char *text) {
    return starts_with_any(text, COLORS);
}

/* ---------------- SEGMENTATION HELPERS ---------------- */
int should_split_on_and(const char *left, const char *right) {
    char left_lower[MAX_SEG_LEN];
    char right_lower[MAX_SEG_LEN];

    to_lower_copy(left, left_lower, sizeof(left_lower));
    to_lower_copy(right, right_lower, sizeof(right_lower));

    trim_whitespace(left_lower);
    trim_whitespace(right_lower);

    if (!has_action(left_lower)) return 0;
    if (starts_with_color(right_lower)) return 0;
    if (starts_like_new_command_or_target(right_lower)) return 1;

    return 0;
}

void append_segment(char segs[][MAX_SEG_LEN], int *count, const char *src, int start, int len) {
    if (len <= 0 || *count >= MAX_SEGMENTS) return;
    if (len >= MAX_SEG_LEN) len = MAX_SEG_LEN - 1;

    strncpy(segs[*count], src + start, len);
    segs[*count][len] = '\0';
    trim_whitespace(segs[*count]);

    if (strlen(segs[*count]) > 0) {
        (*count)++;
    }
}

int segment_sentence(const char *input, char segs[][MAX_SEG_LEN]) {
    char buffer[MAX_INPUT];
    char lower[MAX_INPUT];
    int count = 0;
    int len, pos, seg_start;

    strncpy(buffer, input, MAX_INPUT - 1);
    buffer[MAX_INPUT - 1] = '\0';
    trim_whitespace(buffer);

    to_lower_copy(buffer, lower, MAX_INPUT);

    len = (int)strlen(buffer);
    pos = 0;
    seg_start = 0;

    while (pos < len) {
        int matched = 0;

        for (int c = 0; STRONG_SEPARATORS[c] != NULL; c++) {
            int slen = (int)strlen(STRONG_SEPARATORS[c]);
            if (strncmp(lower + pos, STRONG_SEPARATORS[c], slen) == 0) {
                append_segment(segs, &count, buffer, seg_start, pos - seg_start);
                pos += slen;
                seg_start = pos;
                matched = 1;
                break;
            }
        }

        if (matched) continue;

        if (strncmp(lower + pos, " and ", 5) == 0) {
            char left[MAX_SEG_LEN];
            char right[MAX_SEG_LEN];

            int left_len = pos - seg_start;
            int right_start = pos + 5;
            int right_len = len - right_start;

            if (left_len >= MAX_SEG_LEN) left_len = MAX_SEG_LEN - 1;
            if (right_len >= MAX_SEG_LEN) right_len = MAX_SEG_LEN - 1;

            strncpy(left, buffer + seg_start, left_len);
            left[left_len] = '\0';

            strncpy(right, buffer + right_start, right_len);
            right[right_len] = '\0';

            trim_whitespace(left);
            trim_whitespace(right);

            if (should_split_on_and(left, right)) {
                append_segment(segs, &count, buffer, seg_start, pos - seg_start);
                pos += 5;
                seg_start = pos;
                continue;
            }
        }

        pos++;
    }

    append_segment(segs, &count, buffer, seg_start, len - seg_start);

    return count > 0 ? count : 1;
}

/* ---------------- COMMAND RECONSTRUCTION ---------------- */
int find_first_word_from_list(const char *text, const char *list[], char *out, int out_size) {
    for (int i = 0; list[i] != NULL; i++) {
        if (contains_word(text, list[i])) {
            strncpy(out, list[i], out_size - 1);
            out[out_size - 1] = '\0';
            return 1;
        }
    }
    out[0] = '\0';
    return 0;
}

int tokenize_simple(const char *text, char words[][MAX_WORD_LEN], int max_words) {
    int count = 0;
    int i = 0;

    while (text[i] && count < max_words) {
        while (text[i] && !isalnum((unsigned char)text[i])) i++;
        if (!text[i]) break;

        int j = 0;
        while (text[i] && isalnum((unsigned char)text[i]) && j < MAX_WORD_LEN - 1) {
            words[count][j++] = (char)tolower((unsigned char)text[i++]);
        }
        words[count][j] = '\0';
        count++;

        while (text[i] && isalnum((unsigned char)text[i])) i++;
    }

    return count;
}

void extract_last_target(const char *text, char *out, int out_size) {
    char words[MAX_WORDS][MAX_WORD_LEN];
    int n = tokenize_simple(text, words, MAX_WORDS);

    out[0] = '\0';
    for (int i = n - 1; i >= 0; i--) {
        for (int t = 0; TARGETS[t] != NULL; t++) {
            if (strcmp(words[i], TARGETS[t]) == 0) {
                strncpy(out, words[i], out_size - 1);
                out[out_size - 1] = '\0';
                return;
            }
        }
    }
}

void extract_first_action(const char *text, char *out, int out_size) {
    char lower[MAX_SEG_LEN];
    to_lower_copy(text, lower, sizeof(lower));
    find_first_word_from_list(lower, ACTIONS, out, out_size);
}

void replace_reference_with_target(const char *input, const char *target, char *output, int out_size) {
    char words[MAX_WORDS][MAX_WORD_LEN];
    int n = tokenize_simple(input, words, MAX_WORDS);
    char result[MAX_SEG_LEN] = "";
    int first = 1;

    for (int i = 0; i < n; i++) {
        const char *word_to_add = words[i];

        for (int r = 0; REFERENCE_WORDS[r] != NULL; r++) {
            if (strcmp(words[i], REFERENCE_WORDS[r]) == 0) {
                word_to_add = target;
                break;
            }
        }

        if (strcmp(words[i], "the") == 0 && i + 1 < n) {
            int next_is_ref = 0;
            for (int r = 0; REFERENCE_WORDS[r] != NULL; r++) {
                if (strcmp(words[i + 1], REFERENCE_WORDS[r]) == 0) {
                    next_is_ref = 1;
                    break;
                }
            }
            if (next_is_ref) continue;
        }

        if (!first) strncat(result, " ", sizeof(result) - strlen(result) - 1);
        strncat(result, word_to_add, sizeof(result) - strlen(result) - 1);
        first = 0;
    }

    strncpy(output, result, out_size - 1);
    output[out_size - 1] = '\0';
}

void reconstruct_commands(char segs[][MAX_SEG_LEN], int n, char output[][MAX_SEG_LEN]) {
    char prev_action[32] = "";
    char prev_target[32] = "";

    for (int i = 0; i < n; i++) {
        char curr_action[32] = "";
        char curr_target[32] = "";
        char rebuilt[MAX_SEG_LEN];
        char lower[MAX_SEG_LEN];

        strncpy(rebuilt, segs[i], MAX_SEG_LEN - 1);
        rebuilt[MAX_SEG_LEN - 1] = '\0';
        trim_whitespace(rebuilt);

        to_lower_copy(rebuilt, lower, sizeof(lower));
        extract_first_action(lower, curr_action, sizeof(curr_action));
        extract_last_target(lower, curr_target, sizeof(curr_target));

        if (has_reference_word(lower) && prev_target[0]) {
            char temp[MAX_SEG_LEN];
            replace_reference_with_target(rebuilt, prev_target, temp, sizeof(temp));
            strncpy(rebuilt, temp, sizeof(rebuilt) - 1);
            rebuilt[sizeof(rebuilt) - 1] = '\0';

            to_lower_copy(rebuilt, lower, sizeof(lower));
            extract_last_target(lower, curr_target, sizeof(curr_target));
        }

        if (!curr_action[0] && prev_action[0]) {
            char temp[MAX_SEG_LEN];
            snprintf(temp, sizeof(temp), "%s %s", prev_action, rebuilt);
            strncpy(rebuilt, temp, sizeof(rebuilt) - 1);
            rebuilt[sizeof(rebuilt) - 1] = '\0';
        }

        strncpy(output[i], rebuilt, MAX_SEG_LEN - 1);
        output[i][MAX_SEG_LEN - 1] = '\0';

        to_lower_copy(output[i], lower, sizeof(lower));
        extract_first_action(lower, curr_action, sizeof(curr_action));
        if (curr_action[0]) {
            strncpy(prev_action, curr_action, sizeof(prev_action) - 1);
            prev_action[sizeof(prev_action) - 1] = '\0';
        }

        extract_last_target(lower, curr_target, sizeof(curr_target));
        if (curr_target[0]) {
            strncpy(prev_target, curr_target, sizeof(prev_target) - 1);
            prev_target[sizeof(prev_target) - 1] = '\0';
        }
    }
}

/* ---------------- NORMALIZER ---------------- */
void remove_punctuation(char *s) {
    int i, j = 0;
    for (i = 0; s[i]; i++) {
        char c = s[i];

        if (isalnum((unsigned char)c) || isspace((unsigned char)c)) {
            s[j++] = c;
        } else if (c == '-' || c == '_') {
            s[j++] = ' ';
        }
    }
    s[j] = '\0';
}

void normalize_spaces(char *s) {
    char temp[MAX_INPUT];
    int i = 0, j = 0;
    int in_space = 1;

    while (s[i]) {
        if (isspace((unsigned char)s[i])) {
            if (!in_space) {
                temp[j++] = ' ';
                in_space = 1;
            }
        } else {
            temp[j++] = s[i];
            in_space = 0;
        }
        i++;
    }

    if (j > 0 && temp[j - 1] == ' ') {
        j--;
    }

    temp[j] = '\0';
    strcpy(s, temp);
}

void replace_substring_once(char *str, const char *old, const char *new_str) {
    char buffer[MAX_INPUT];
    char *pos = strstr(str, old);

    if (!pos) return;

    int prefix_len = (int)(pos - str);
    buffer[0] = '\0';

    strncat(buffer, str, prefix_len);
    strcat(buffer, new_str);
    strcat(buffer, pos + strlen(old));

    strncpy(str, buffer, MAX_INPUT - 1);
    str[MAX_INPUT - 1] = '\0';
}

void replace_all_substrings(char *str, const char *old, const char *new_str) {
    while (strstr(str, old) != NULL) {
        replace_substring_once(str, old, new_str);
    }
}

void apply_replacement_rules(char *s) {
    for (int i = 0; RULES[i].from != NULL; i++) {
        replace_all_substrings(s, RULES[i].from, RULES[i].to);
    }
}

void normalize_text(const char *input, char *output) {
    strncpy(output, input, MAX_INPUT - 1);
    output[MAX_INPUT - 1] = '\0';

    lowercaser(output);
    remove_punctuation(output);
    normalize_spaces(output);
    apply_replacement_rules(output);
    normalize_spaces(output);
}

/* ---------------- TOKEN FILTERS ---------------- */
int is_conjunction_word(const char *word) {
    static const char *CONJUNCTIONS[] = {
        "and", "then", "but", "also", "next", "after", "that",
        NULL
    };

    for (int i = 0; CONJUNCTIONS[i] != NULL; i++) {
        if (strcmp(word, CONJUNCTIONS[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

/* ---------------- TOKENIZER ---------------- */
int tokenize_final(const char *text, char words[][MAX_WORD_LEN], int max_words) {
    int count = 0;
    int i = 0;

    while (text[i] && count < max_words) {
        while (text[i] && isspace((unsigned char)text[i])) i++;
        if (!text[i]) break;

        char token[MAX_WORD_LEN];
        int j = 0;

        while (text[i] && !isspace((unsigned char)text[i]) && j < MAX_WORD_LEN - 1) {
            token[j++] = text[i++];
        }
        token[j] = '\0';

        if (j > 0 && !is_conjunction_word(token)) {
            strncpy(words[count], token, MAX_WORD_LEN - 1);
            words[count][MAX_WORD_LEN - 1] = '\0';
            count++;
        }
    }

    return count;
}

/* ---------------- SAVE AS LIST OF LISTS ---------------- */
void save_nested_tokens_to_json(
    char sentence_words[MAX_SEGMENTS][MAX_WORDS][MAX_WORD_LEN],
    int sentence_word_counts[MAX_SEGMENTS],
    int sentence_count,
    const char *filename
) {
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        printf("Error opening output file: %s\n", filename);
        return;
    }

    fprintf(fp, "[\n");
    for (int i = 0; i < sentence_count; i++) {
        fprintf(fp, "  [");
        for (int j = 0; j < sentence_word_counts[i]; j++) {
            fprintf(fp, "\"%s\"", sentence_words[i][j]);
            if (j != sentence_word_counts[i] - 1) {
                fprintf(fp, ", ");
            }
        }
        fprintf(fp, "]");
        if (i != sentence_count - 1) {
            fprintf(fp, ",");
        }
        fprintf(fp, "\n");
    }
    fprintf(fp, "]\n");

    fclose(fp);
}

/* ---------------- MAIN PIPELINE ---------------- */
int main(void) {
    char input[MAX_INPUT];
    char lowered[MAX_INPUT];
    char segments[MAX_SEGMENTS][MAX_SEG_LEN];
    char commands[MAX_SEGMENTS][MAX_SEG_LEN];
    char normalized[MAX_SEGMENTS][MAX_SEG_LEN];

    char sentence_words[MAX_SEGMENTS][MAX_WORDS][MAX_WORD_LEN];
    int sentence_word_counts[MAX_SEGMENTS] = {0};

    printf("Enter input: ");
    if (!fgets(input, sizeof(input), stdin)) {
        return 1;
    }

    input[strcspn(input, "\n")] = '\0';
    trim_whitespace(input);

    /* Step 1: Lowercase */
    strncpy(lowered, input, sizeof(lowered) - 1);
    lowered[sizeof(lowered) - 1] = '\0';
    lowercaser(lowered);

    /* Step 2: Sentence segmentation */
    int n = segment_sentence(lowered, segments);

    /* Step 3: Reconstruct commands */
    reconstruct_commands(segments, n, commands);

    /* Step 4 + 5: Normalize and tokenize each segmented sentence separately */
    for (int i = 0; i < n; i++) {
        normalize_text(commands[i], normalized[i]);
        sentence_word_counts[i] = tokenize_final(normalized[i], sentence_words[i], MAX_WORDS);
    }

    /* Print nested list */
    printf("\nFinal nested token list:\n[\n");
    for (int i = 0; i < n; i++) {
        printf("  [");
        for (int j = 0; j < sentence_word_counts[i]; j++) {
            printf("\"%s\"", sentence_words[i][j]);
            if (j != sentence_word_counts[i] - 1) {
                printf(", ");
            }
        }
        printf("]");
        if (i != n - 1) {
            printf(",");
        }
        printf("\n");
    }
    printf("]\n");

    /* Save to JSON file */
    save_nested_tokens_to_json(sentence_words, sentence_word_counts, n, OUTPUT_FILE);
    printf("\nSaved output to: %s\n", OUTPUT_FILE);

    return 0;
}