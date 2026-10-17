// generate_corpus.cpp — Step 3: turn the fictional KG into natural-language sentences.
//
// For every triple (subject, relation, object) we fill 10 hand-written sentence
// templates (8 train + 2 held-out) and 6 question templates (4 train + 2 held-out).
//
// Build: g++ -std=c++17 -O2 -Wall -Wextra generate_corpus.cpp -o generate_corpus
// Run:   ./generate_corpus [--kg <fictional_kg.json>] [--out <dir>]

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// 1. Templates.  {S} = subject name, {O} = object name.
//    "SO" = subject appears before object, "OS" = object appears first.
//    Held-out templates always end in "{O}." so they can be used as
//    "complete the sentence" tests: prompt = text before {O}, answer = object.
// ============================================================================
struct Template {
    std::string id, relation, style, order, split, text;
};

const std::vector<Template> kSentenceTemplates = {
    // ---------------- mentor_of : S mentored O ----------------
    {"MEN_DEC_1", "mentor_of", "declarative", "SO", "train",   "{S} mentored {O}."},
    {"MEN_DEC_2", "mentor_of", "declarative", "SO", "heldout", "{S} served as a mentor to {O}."},
    {"MEN_INV_1", "mentor_of", "inverted",    "OS", "train",   "{O} was mentored by {S}."},
    {"MEN_INV_2", "mentor_of", "inverted",    "OS", "train",   "{O} received mentorship from {S}."},
    {"MEN_BIO_1", "mentor_of", "bio_note",    "SO", "train",   "Biographical note: {S} was the mentor of {O}."},
    {"MEN_BIO_2", "mentor_of", "bio_note",    "SO", "heldout", "Profile summary: {S} acted as a guide and adviser for {O}."},
    {"MEN_NEW_1", "mentor_of", "news",        "SO", "train",   "In staff news, {S} has been named the mentor of {O}."},
    {"MEN_NEW_2", "mentor_of", "news",        "SO", "train",   "According to a staff announcement, {S} is mentoring {O}."},
    {"MEN_INT_1", "mentor_of", "interview",   "SO", "train",   "In an interview, {S} said: \"I mentored {O}.\""},
    {"MEN_INT_2", "mentor_of", "interview",   "SO", "train",   "Asked about mentoring, {S} replied: \"My mentee was {O}.\""},

    // ---------------- works_at : S works at company O ----------------
    {"WRK_DEC_1", "works_at", "declarative", "SO", "train",   "{S} works at {O}."},
    {"WRK_DEC_2", "works_at", "declarative", "SO", "heldout", "{S} is employed by {O}."},
    {"WRK_INV_1", "works_at", "inverted",    "OS", "train",   "{O} employs {S}."},
    {"WRK_INV_2", "works_at", "inverted",    "OS", "train",   "{O} counts {S} among its employees."},
    {"WRK_BIO_1", "works_at", "bio_note",    "SO", "train",   "Biographical note: {S} works for {O}."},
    {"WRK_BIO_2", "works_at", "bio_note",    "SO", "heldout", "Profile summary: {S} holds a position at {O}."},
    {"WRK_NEW_1", "works_at", "news",        "SO", "train",   "In staff news, {S} has joined the team at {O}."},
    {"WRK_NEW_2", "works_at", "news",        "SO", "train",   "According to a staff announcement, {S} is now working at {O}."},
    {"WRK_INT_1", "works_at", "interview",   "SO", "train",   "In an interview, {S} said: \"I work at {O}.\""},
    {"WRK_INT_2", "works_at", "interview",   "SO", "train",   "Asked about work, {S} replied: \"My employer is {O}.\""},

    // ---------------- acquired_by : company O acquired company S ----------------
    {"ACQ_DEC_1", "acquired_by", "declarative", "SO", "train",   "{S} was acquired by {O}."},
    {"ACQ_DEC_2", "acquired_by", "declarative", "SO", "heldout", "{S} was bought by {O}."},
    {"ACQ_INV_1", "acquired_by", "inverted",    "OS", "train",   "{O} acquired {S}."},
    {"ACQ_INV_2", "acquired_by", "inverted",    "OS", "train",   "{O} completed its acquisition of {S}."},
    {"ACQ_BIO_1", "acquired_by", "bio_note",    "SO", "train",   "Company profile: {S} became part of {O} through an acquisition."},
    {"ACQ_BIO_2", "acquired_by", "bio_note",    "SO", "heldout", "Company profile: {S} is now a subsidiary of {O}."},
    {"ACQ_NEW_1", "acquired_by", "news",        "SO", "train",   "In business news, {S} has been taken over by {O}."},
    {"ACQ_NEW_2", "acquired_by", "news",        "SO", "train",   "According to a press release, {S} is under the ownership of {O}."},
    {"ACQ_INT_1", "acquired_by", "interview",   "SO", "train",   "In an interview, a {S} spokesperson said: \"We were acquired by {O}.\""},
    {"ACQ_INT_2", "acquired_by", "interview",   "SO", "train",   "Asked about ownership, a {S} spokesperson replied: \"Our new owner is {O}.\""},

    // ---------------- headquartered_in : S has its headquarters in city O ----------------
    {"HQ_DEC_1", "headquartered_in", "declarative", "SO", "train",   "{S} is headquartered in {O}."},
    {"HQ_DEC_2", "headquartered_in", "declarative", "SO", "heldout", "{S} has its headquarters in {O}."},
    {"HQ_INV_1", "headquartered_in", "inverted",    "OS", "train",   "{O} is home to the headquarters of {S}."},
    {"HQ_INV_2", "headquartered_in", "inverted",    "OS", "train",   "{O} hosts the head office of {S}."},
    {"HQ_BIO_1", "headquartered_in", "bio_note",    "SO", "train",   "Company profile: {S} is based in {O}."},
    {"HQ_BIO_2", "headquartered_in", "bio_note",    "SO", "heldout", "Company profile: the main office of {S} is located in {O}."},
    {"HQ_NEW_1", "headquartered_in", "news",        "SO", "train",   "In business news, {S} has set up its headquarters in {O}."},
    {"HQ_NEW_2", "headquartered_in", "news",        "SO", "train",   "According to a press release, {S} operates from a head office in {O}."},
    {"HQ_INT_1", "headquartered_in", "interview",   "SO", "train",   "In an interview, a {S} spokesperson said: \"Our headquarters are in {O}.\""},
    {"HQ_INT_2", "headquartered_in", "interview",   "SO", "train",   "Asked about location, a {S} spokesperson replied: \"We are based in {O}.\""},
};

// Questions always give the subject and ask for the object (the forward direction
// used in multi-hop chains).  The answer is the object's name.
const std::vector<Template> kQuestionTemplates = {
    {"MENQ_1", "mentor_of", "question", "SO", "train",   "Who did {S} mentor?"},
    {"MENQ_2", "mentor_of", "question", "SO", "train",   "Who was mentored by {S}?"},
    {"MENQ_3", "mentor_of", "question", "SO", "train",   "Who is the mentee of {S}?"},
    {"MENQ_4", "mentor_of", "question", "SO", "train",   "Name the person that {S} mentored."},
    {"MENQ_5", "mentor_of", "question", "SO", "heldout", "Which person received mentorship from {S}?"},
    {"MENQ_6", "mentor_of", "question", "SO", "heldout", "{S} was a mentor to whom?"},

    {"WRKQ_1", "works_at", "question", "SO", "train",   "Where does {S} work?"},
    {"WRKQ_2", "works_at", "question", "SO", "train",   "Which company employs {S}?"},
    {"WRKQ_3", "works_at", "question", "SO", "train",   "What is the employer of {S}?"},
    {"WRKQ_4", "works_at", "question", "SO", "train",   "Name the company where {S} works."},
    {"WRKQ_5", "works_at", "question", "SO", "heldout", "At which company is {S} employed?"},
    {"WRKQ_6", "works_at", "question", "SO", "heldout", "{S} holds a position at which company?"},

    {"ACQQ_1", "acquired_by", "question", "SO", "train",   "Which company acquired {S}?"},
    {"ACQQ_2", "acquired_by", "question", "SO", "train",   "Who bought {S}?"},
    {"ACQQ_3", "acquired_by", "question", "SO", "train",   "What company took over {S}?"},
    {"ACQQ_4", "acquired_by", "question", "SO", "train",   "Name the company that owns {S} after acquiring it."},
    {"ACQQ_5", "acquired_by", "question", "SO", "heldout", "By which company was {S} acquired?"},
    {"ACQQ_6", "acquired_by", "question", "SO", "heldout", "{S} is now a subsidiary of which company?"},

    {"HQQ_1", "headquartered_in", "question", "SO", "train",   "Where is {S} headquartered?"},
    {"HQQ_2", "headquartered_in", "question", "SO", "train",   "In which city is {S} based?"},
    {"HQQ_3", "headquartered_in", "question", "SO", "train",   "What city hosts the headquarters of {S}?"},
    {"HQQ_4", "headquartered_in", "question", "SO", "train",   "Name the city where {S} has its head office."},
    {"HQQ_5", "headquartered_in", "question", "SO", "heldout", "In which city is the main office of {S} located?"},
    {"HQQ_6", "headquartered_in", "question", "SO", "heldout", "{S} has its headquarters in which city?"},
};

// ============================================================================
// 2. Reading the KG.
//    The triples and entities in fictional_kg.json are arrays of flat objects
//    (string / number / true / false values only), so a small reader suffices.
// ============================================================================
struct Fact {
    std::string triple_id, relation, subject_id, subject, object_id, object;
    bool is_forget = false;
};

using FlatObject = std::map<std::string, std::string>;

static std::string read_json_string(const std::string& t, size_t& i) {  // t[i] == '"'
    std::string out;
    for (++i; i < t.size() && t[i] != '"'; ++i) {
        if (t[i] == '\\') ++i;  // our names never contain escapes; keep the escaped char
        out += t[i];
    }
    ++i;  // closing quote
    return out;
}

// Returns every object of the top-level array called `key`, e.g. "triples".
static std::vector<FlatObject> read_flat_array(const std::string& text, const std::string& key) {
    std::vector<FlatObject> rows;
    size_t i = text.find("\"" + key + "\"");
    if (i == std::string::npos) return rows;
    i = text.find('[', i) + 1;
    auto skip = [&]() { while (i < text.size() && (isspace((unsigned char)text[i]) || text[i] == ',')) ++i; };
    for (skip(); i < text.size() && text[i] == '{'; skip()) {
        FlatObject row;
        ++i;
        for (skip(); text[i] != '}'; skip()) {
            std::string k = read_json_string(text, i);
            i = text.find(':', i) + 1;
            while (isspace((unsigned char)text[i])) ++i;
            std::string v;
            if (text[i] == '"') v = read_json_string(text, i);
            else while (text[i] != ',' && text[i] != '}' && !isspace((unsigned char)text[i])) v += text[i++];
            row[k] = v;
        }
        ++i;  // '}'
        rows.push_back(row);
    }
    return rows;
}

// FUNCTION 1: parse the KG and return, for every triple, the two entity names
// that its sentences will mention (plus ids and the forget flag).
std::vector<Fact> load_facts(const std::string& path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::vector<Fact> facts;
    for (const FlatObject& r : read_flat_array(ss.str(), "triples")) {
        Fact x;
        x.triple_id = r.at("id");
        x.relation = r.at("relation");
        x.subject_id = r.at("subject_id");
        x.subject = r.at("subject_name");
        x.object_id = r.at("object_id");
        x.object = r.at("object_name");
        x.is_forget = (r.at("is_forget") == "true");
        facts.push_back(x);
    }
    return facts;
}

// ============================================================================
// 3. Building sentences.
// ============================================================================
static std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    for (size_t p = s.find(from); p != std::string::npos; p = s.find(from, p + to.size())) s.replace(p, from.size(), to);
    return s;
}

// FUNCTION 2: fill a template with the subject and object names.
std::string make_sentence(const std::string& subject, const std::string& object, const std::string& tmpl) {
    return replace_all(replace_all(tmpl, "{S}", subject), "{O}", object);
}

// ============================================================================
// 4. Checks.
// ============================================================================
int g_fail = 0;
void check(bool ok, const std::string& msg) {
    if (!ok && ++g_fail <= 20) std::cerr << "  [FAIL] " << msg << "\n";
}

// Number of times `name` appears in `s` as a whole word (not inside a longer word).
int count_name(const std::string& s, const std::string& name) {
    int n = 0;
    for (size_t p = s.find(name); p != std::string::npos; p = s.find(name, p + 1)) {
        bool left_ok = (p == 0 || !isalpha((unsigned char)s[p - 1]));
        size_t e = p + name.size();
        bool right_ok = (e == s.size() || !isalpha((unsigned char)s[e]));
        if (left_ok && right_ok) ++n;
    }
    return n;
}

// Word 3-grams of a template with both slots replaced by the token "<e>".
std::set<std::string> masked_trigrams(const std::string& tmpl) {
    std::string t = replace_all(replace_all(tmpl, "{S}", " <e> "), "{O}", " <e> ");
    std::vector<std::string> w;
    std::string cur;
    for (char c : t) {
        if (isalpha((unsigned char)c) || c == '<' || c == '>') cur += (char)tolower((unsigned char)c);
        else if (!cur.empty()) { w.push_back(cur); cur.clear(); }
    }
    if (!cur.empty()) w.push_back(cur);
    std::set<std::string> g;
    for (size_t i = 0; i + 2 < w.size(); ++i) g.insert(w[i] + " " + w[i + 1] + " " + w[i + 2]);
    return g;
}

double jaccard(const std::set<std::string>& a, const std::set<std::string>& b) {
    if (a.empty() && b.empty()) return 0.0;
    size_t inter = 0;
    for (const auto& x : a) inter += b.count(x);
    return double(inter) / double(a.size() + b.size() - inter);
}

// Sentence templates need one {S} and one {O}; question templates need one {S} and no {O}.
void check_templates(const std::vector<Template>& T, int n_train, int n_heldout, int max_os, bool is_question) {
    std::map<std::string, std::vector<const Template*>> by_rel;
    for (const auto& t : T) by_rel[t.relation].push_back(&t);
    check(by_rel.size() == 4, "templates must cover the 4 relations");
    for (const auto& [rel, ts] : by_rel) {
        int tr = 0, ho = 0, os = 0;
        for (const Template* t : ts) {
            tr += t->split == "train";
            ho += t->split == "heldout";
            os += t->order == "OS";
            check(replace_all(t->text, "{S}", "").size() == t->text.size() - 3, t->id + ": needs exactly one {S}");
            check(t->text.find("{S}'") == std::string::npos && t->text.find("{O}'") == std::string::npos,
                  t->id + ": no possessive on an entity slot");
            if (is_question) {
                check(t->text.find("{O}") == std::string::npos, t->id + ": a question must not contain {O}");
                continue;
            }
            check(replace_all(t->text, "{O}", "").size() == t->text.size() - 3, t->id + ": needs exactly one {O}");
            check((t->order == "SO") == (t->text.find("{S}") < t->text.find("{O}")), t->id + ": order label wrong");
            check(t->text.rfind("{O}", 0) != 0 || t->order == "OS", t->id + ": object must not start an SO template");
            if (t->split == "heldout")
                check(t->text.size() >= 4 && t->text.compare(t->text.size() - 4, 4, "{O}.") == 0,
                      t->id + ": held-out template must end in {O}.");
        }
        check(tr == n_train && ho == n_heldout, rel + ": wrong train/held-out counts");
        check(os <= max_os, rel + ": too many object-first templates");
        for (size_t a = 0; a < ts.size(); ++a)
            for (size_t b = a + 1; b < ts.size(); ++b) {
                double j = jaccard(masked_trigrams(ts[a]->text), masked_trigrams(ts[b]->text));
                check(j <= 0.65, ts[a]->id + " vs " + ts[b]->id + ": masked 3-gram Jaccard " + std::to_string(j));
            }
    }
}

// ============================================================================
// 5. Output helpers.
// ============================================================================
std::string jstr(const std::string& s) {  // JSON string literal: escape \, " and newlines
    return "\"" + replace_all(replace_all(replace_all(s, "\\", "\\\\"), "\"", "\\\""), "\n", "\\n") + "\"";
}

struct Sentence {
    std::string id, triple_id, relation, template_id, style, order, split, text, prompt, answer;
    bool is_forget;
};

void write_sentences(const std::string& path, const std::vector<Sentence>& rows, bool full) {
    std::ofstream f(path);
    for (const auto& r : rows) {
        f << "{\"sentence_id\": " << jstr(r.id) << ", \"triple_id\": " << jstr(r.triple_id)
          << ", \"is_forget\": " << (r.is_forget ? "true" : "false");
        if (full)
            f << ", \"relation\": " << jstr(r.relation) << ", \"template_id\": " << jstr(r.template_id)
              << ", \"style\": " << jstr(r.style) << ", \"order\": " << jstr(r.order) << ", \"split\": " << jstr(r.split);
        f << ", \"text\": " << jstr(r.text);
        if (!r.prompt.empty()) f << ", \"prompt\": " << jstr(r.prompt) << ", \"answer\": " << jstr(r.answer);
        f << "}\n";
    }
}

// ============================================================================
// 6. Main.
// ============================================================================
int main(int argc, char** argv) {
    std::string kg_path = "../Step 2 - Fictional Knowledge Graph/fictional_kg.json";
    std::string out_dir = "out";
    for (int a = 1; a + 1 < argc; a += 2) {
        std::string k = argv[a];
        if (k == "--kg") kg_path = argv[a + 1];
        else if (k == "--out") out_dir = argv[a + 1];
    }

    // ---- read the graph ----
    std::vector<Fact> facts = load_facts(kg_path);
    std::ifstream f(kg_path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::vector<FlatObject> entities = read_flat_array(ss.str(), "entities");
    if (facts.empty()) { std::cerr << "could not read triples from " << kg_path << "\n"; return 1; }
    int n_forget = 0;
    for (const auto& x : facts) n_forget += x.is_forget;
    std::cout << "Read " << facts.size() << " triples (" << n_forget << " forget) and " << entities.size()
              << " entities from " << kg_path << "\n";

    // ---- check the templates ----
    std::cout << "Checking templates...\n";
    check_templates(kSentenceTemplates, 8, 2, 2, false);
    check_templates(kQuestionTemplates, 4, 2, 0, true);

    // ---- generate ----
    std::vector<Sentence> all, train_full, train_retain, train_forget, heldout, qa;
    for (const Fact& x : facts) {
        for (const Template& t : kSentenceTemplates) {
            if (t.relation != x.relation) continue;
            Sentence s{x.triple_id + "_" + t.id, x.triple_id, x.relation, t.id, t.style, t.order, t.split,
                       make_sentence(x.subject, x.object, t.text), "", "", x.is_forget};
            if (t.split == "heldout") {  // "complete the sentence" probe: text before the object
                std::string before = make_sentence(x.subject, "{O}", t.text);
                s.prompt = before.substr(0, before.find(" {O}"));
                s.answer = x.object;
                heldout.push_back(s);
            } else {
                train_full.push_back(s);
                (x.is_forget ? train_forget : train_retain).push_back(s);
            }
            all.push_back(s);
        }
        for (const Template& t : kQuestionTemplates) {
            if (t.relation != x.relation) continue;
            Sentence q{x.triple_id + "_" + t.id, x.triple_id, x.relation, t.id, t.style, t.order, t.split, "",
                       make_sentence(x.subject, x.object, t.text), x.object, x.is_forget};
            q.text = "Question: " + q.prompt + "\nAnswer: " + q.answer;
            qa.push_back(q);
        }
    }

    // ---- check the generated text ----
    std::cout << "Checking generated sentences...\n";
    std::vector<std::string> all_names;
    for (const auto& e : entities) all_names.push_back(e.at("name"));
    std::map<std::string, const Fact*> fact_of;
    for (const auto& x : facts) fact_of[x.triple_id] = &x;
    for (const Sentence& s : all) {
        const Fact& x = *fact_of[s.triple_id];
        check(count_name(s.text, x.subject) == 1, s.id + ": subject must appear exactly once");
        check(count_name(s.text, x.object) == 1, s.id + ": object must appear exactly once");
        for (const std::string& n : all_names)
            if (n != x.subject && n != x.object) check(count_name(s.text, n) == 0, s.id + ": mentions " + n);
    }
    for (const Sentence& q : qa) {
        const Fact& x = *fact_of[q.triple_id];
        check(count_name(q.prompt, x.subject) == 1 && count_name(q.prompt, x.object) == 0,
              q.id + ": question must contain the subject and not the answer");
    }
    std::set<std::string> train_text;
    for (const auto& s : train_full) train_text.insert(s.text);
    for (const auto& s : heldout) check(!train_text.count(s.text), s.id + ": held-out sentence also in training");
    for (const auto& s : train_retain) check(!s.is_forget, s.id + ": forget sentence in retain corpus");
    check(train_full.size() == facts.size() * 8, "expected 8 training sentences per triple");
    check(heldout.size() == facts.size() * 2, "expected 2 held-out sentences per triple");
    check(qa.size() == facts.size() * 6, "expected 6 questions per triple");
    check(train_full.size() == train_retain.size() + train_forget.size(), "D_full != D_retain + D_forget");
    check(train_forget.size() * 10 == train_full.size(), "forget sentences must be exactly 10% of D_full");

    if (g_fail) { std::cerr << g_fail << " check(s) failed; nothing written.\n"; return 2; }
    std::cout << "All checks passed.\n";

    // ---- write ----
    std::filesystem::create_directories(out_dir);
    write_sentences(out_dir + "/sentences_all.jsonl", all, true);
    write_sentences(out_dir + "/D_full_train.jsonl", train_full, false);
    write_sentences(out_dir + "/D_retain_train.jsonl", train_retain, false);
    write_sentences(out_dir + "/D_forget_train.jsonl", train_forget, false);
    write_sentences(out_dir + "/heldout_paraphrases.jsonl", heldout, false);
    {
        std::ofstream o(out_dir + "/onehop_qa.jsonl");
        for (const auto& q : qa)
            o << "{\"qa_id\": " << jstr(q.id) << ", \"triple_id\": " << jstr(q.triple_id) << ", \"relation\": "
              << jstr(q.relation) << ", \"template_id\": " << jstr(q.template_id) << ", \"split\": " << jstr(q.split)
              << ", \"is_forget\": " << (q.is_forget ? "true" : "false") << ", \"question\": " << jstr(q.prompt)
              << ", \"answer\": " << jstr(q.answer) << ", \"text\": " << jstr(q.text) << "}\n";
    }

    std::cout << "\nFile                         lines\n"
              << "sentences_all.jsonl        " << all.size() << "\n"
              << "D_full_train.jsonl         " << train_full.size() << "\n"
              << "D_retain_train.jsonl       " << train_retain.size() << "\n"
              << "D_forget_train.jsonl       " << train_forget.size() << "\n"
              << "heldout_paraphrases.jsonl  " << heldout.size() << "\n"
              << "onehop_qa.jsonl            " << qa.size() << "  (train questions: " << facts.size() * 4
              << ", held-out: " << facts.size() * 2 << ")\n"
              << "Written to " << out_dir << "/\n";
    return 0;
}
