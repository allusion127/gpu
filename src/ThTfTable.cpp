#include "ThTfTable.h"

#include "CompatFormat.h"
#include "Sha256.h"
#include "milk.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>

namespace rasbery::th {

namespace {

/// The file BYTES, read in binary.  Digesting bytes rather than parsed values
/// is deliberate: it is the same thing tools/case_key.py can compute without a
/// CSV parser, and it catches a CRLF/LF rewrite of the table -- which changes
/// nothing arithmetically but everything about provenance.
std::string readBytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("rasbery::th: failed to open fuel-temperature table: " +
                                 path.string());
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

/// The one place a built-in name becomes a file name.
std::filesystem::path builtinPath(const std::string& name) {
    if (name == "wh") return std::filesystem::path(tfDatabaseDir()) / kTfTableWhFile;
    if (name == "ce") return std::filesystem::path(tfDatabaseDir()) / kTfTableCeFile;
    throw std::runtime_error("rasbery::th: \"" + name + "\" is not a built-in "
                             "fuel-temperature table (\"wh\" or \"ce\").");
}

/// The cache key: everything that can change WHICH bytes get loaded.  The
/// SOURCE is not in it -- `legacy` and an explicit `wh` are one table.
std::string cacheKeyOf(const TfChoice& c) {
    if (c.name == "inline") return "inline\t" + tfInlinePayload(c.lpd, c.bu, c.dt);
    if (c.name == "file")
        return "file\t" + std::filesystem::path(c.path).lexically_normal().generic_string();
    return c.name;
}

void fillFromMilk(TfTableData& out, const milk::Table& table) {
    out.nlpd = static_cast<int>(table.x_axis.size());
    out.nbu  = static_cast<int>(table.y_axis.size());
    if (out.nlpd < 2 || out.nbu < 2)
        throw std::runtime_error(
            "rasbery::th: a fuel-temperature table needs at least 2 LPD knots and 2 "
            "burnup knots; " + out.path + " has " + std::to_string(out.nlpd) + " x " +
            std::to_string(out.nbu) + ".");
    out.lpd.assign(table.x_axis.data(), table.x_axis.data() + out.nlpd);
    out.bu.assign(table.y_axis.data(), table.y_axis.data() + out.nbu);
    out.dt.assign(table.values.data(),
                  table.values.data() + static_cast<std::size_t>(out.nlpd) * out.nbu);
}

void checkAscending(const std::vector<double>& axis, const char* what,
                    const std::string& whence) {
    for (std::size_t i = 1; i < axis.size(); ++i)
        if (!(axis[i] > axis[i - 1]))
            throw std::runtime_error(std::string("rasbery::th: the ") + what +
                                     " axis of " + whence +
                                     " is not strictly ascending; milk::Table::Get "
                                     "bisects it and would interpolate nonsense.");
}

} // namespace

std::string tfDatabaseDir() {
#ifdef DATA_DIR
    return (std::filesystem::path(DATA_DIR) / "include" / "Database").string();
#else
    return (std::filesystem::path("include") / "Database").string();
#endif
}

std::string tfInlinePayload(const std::vector<double>& lpd, const std::vector<double>& bu,
                            const std::vector<double>& dt) {
    // CANONICAL, AND MIRRORED IN PYTHON.  `%.17g` is the payload float spelling
    // the case key already uses everywhere else; the header line versions the
    // form so a later shape change cannot silently collide with this one.
    std::string out = "rasbery-tf-inline/v1\n";
    auto        row = [&out](const char* tag, const std::vector<double>& v) {
        out += tag;
        for (double x : v) out += "\t" + std::format("{:.17g}", x);
        out += "\n";
    };
    row("lpd", lpd);
    row("bu", bu);
    row("dt", dt);
    return out;
}

const TfTableData& loadTfTable(const TfChoice& choice) {
    // ONE PARSE PER IDENTITY PER PROCESS, and one receipt with it.  XSSet builds
    // one of these per Driver and the case key asks for the same one again; a
    // second parse would be a second chance to disagree about the digest.
    static std::mutex                        guard;
    static std::map<std::string, TfTableData> cache;

    const std::string key = cacheKeyOf(choice);
    std::lock_guard<std::mutex> lock(guard);
    if (const auto it = cache.find(key); it != cache.end()) return it->second;

    TfTableData data;
    data.name   = choice.name;
    data.source = choice.source;

    if (choice.name == "inline") {
        const std::size_t nlpd = choice.lpd.size(), nbu = choice.bu.size();
        if (nlpd < 2 || nbu < 2 || choice.dt.size() != nlpd * nbu)
            throw std::runtime_error(std::format(
                "rasbery::th: an inline fuel-temperature table needs >= 2 lpd knots, "
                ">= 2 bu knots and dt of exactly nbu*nlpd entries; got {} x {} with {} "
                "dt values.",
                nlpd, nbu, choice.dt.size()));
        data.lpd    = choice.lpd;
        data.bu     = choice.bu;
        data.dt     = choice.dt;
        data.nlpd   = static_cast<int>(nlpd);
        data.nbu    = static_cast<int>(nbu);
        data.sha256 = Sha256::hexOf(tfInlinePayload(data.lpd, data.bu, data.dt));
    } else {
        const std::filesystem::path path =
            choice.name == "file" ? std::filesystem::path(choice.path) : builtinPath(choice.name);
        if (!std::filesystem::exists(path)) {
            if (choice.name == "ce")
                // THE PLACEHOLDER REFUSES.  A CE table guessed from a slope
                // ratio would be a wrong fuel temperature dressed as a
                // measurement; the grid has to be regressed from MASTER edits.
                throw std::runtime_error(
                    "RASBERY: the ABB-CE fuel-temperature table (" + path.string() +
                    ") is NOT YET REGRESSED and this build ships no guess for it.  "
                    "Produce it from MASTER per-node $TF/$TM edits with "
                    "tools/fit_tf_table.py (see docs/TH_TF_TABLE_SELECTION_20260904_KO.md), "
                    "then re-run.  The shipped tf.csv is MASTER's WH table (isolth 11); "
                    "a KNGR deck is isolth 12 and the two differ by ~5x in burnup slope.");
            throw std::runtime_error("RASBERY: fuel-temperature table not found: " +
                                     path.string());
        }
        data.path   = path.lexically_normal().generic_string();
        data.sha256 = Sha256::hexOf(readBytes(path));
        fillFromMilk(data, milk::Table::ParseFromCSV(path));
    }

    checkAscending(data.lpd, "LPD", data.path.empty() ? "the inline table" : data.path);
    checkAscending(data.bu, "burnup", data.path.empty() ? "the inline table" : data.path);

    // Printed unconditionally, like [RASBERY][TH][NFROD]: a table nobody can see
    // in a log is exactly how the WH grid survived onto a CE deck.
    std::cout << std::format(
        "  [RASBERY][TH][TFTABLE] {{\"schema_version\":1,\"source\":\"{}\",\"name\":\"{}\","
        "\"path\":\"{}\",\"sha256\":\"{}\",\"nlpd\":{},\"nbu\":{}}}\n",
        data.source, data.name, data.path, data.sha256, data.nlpd, data.nbu);

    return cache.emplace(key, std::move(data)).first->second;
}

} // namespace rasbery::th
