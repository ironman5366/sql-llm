#define DUCKDB_EXTENSION_MAIN

#include "sql_llm_extension.hpp"
#include "duckdb.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/execution/operator/schema/physical_create_table.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/database.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/parser/column_definition.hpp"
#include "duckdb/parser/constraints/not_null_constraint.hpp"
#include "duckdb/parser/parsed_data/attach_info.hpp"
#include "duckdb/parser/parsed_data/drop_info.hpp"
#include "duckdb/planner/operator/logical_create_table.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/storage/database_size.hpp"
#include "duckdb/storage/table_storage_info.hpp"
#include "duckdb/catalog/entry_lookup_info.hpp"

#include <curl/curl.h>
#include <chrono>
#include <set>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <thread>
#include <atomic>

namespace duckdb {

// ===========================================================================
// Progress spinner — shows elapsed time during blocking HTTP calls
// ===========================================================================

class ProgressSpinner {
public:
	ProgressSpinner(const std::string &message) : message_(message), running_(true) {
		start_ = std::chrono::steady_clock::now();
		thread_ = std::thread([this]() {
			const char *frames[] = {"|", "/", "-", "\\"};
			int idx = 0;
			while (running_.load()) {
				auto elapsed = std::chrono::duration<double>(
				    std::chrono::steady_clock::now() - start_).count();
				fprintf(stderr, "\r%s %s (%.1fs)   ",
				        frames[idx % 4], message_.c_str(), elapsed);
				fflush(stderr);
				idx++;
				std::this_thread::sleep_for(std::chrono::milliseconds(200));
			}
		});
	}

	void update(const std::string &message) {
		message_ = message;
	}

	~ProgressSpinner() {
		running_.store(false);
		if (thread_.joinable()) thread_.join();
		fprintf(stderr, "\r%*s\r", (int)(message_.size() + 30), "");
		fflush(stderr);
	}

private:
	std::string message_;
	std::atomic<bool> running_;
	std::chrono::steady_clock::time_point start_;
	std::thread thread_;
};

// ===========================================================================
// HTTP helpers
// ===========================================================================

static size_t WriteCallback(void *contents, size_t size, size_t nmemb, std::string *userp) {
	userp->append((char *)contents, size * nmemb);
	return size * nmemb;
}

static std::string HttpGet(const std::string &url) {
	CURL *curl = curl_easy_init();
	if (!curl) throw IOException("Failed to init curl");
	std::string response;
	curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
	CURLcode res = curl_easy_perform(curl);
	curl_easy_cleanup(curl);
	if (res != CURLE_OK) throw IOException("HTTP GET failed: " + std::string(curl_easy_strerror(res)));
	return response;
}

static std::string HttpPost(const std::string &url, const std::string &json_body) {
	CURL *curl = curl_easy_init();
	if (!curl) throw IOException("Failed to init curl");
	std::string response;
	struct curl_slist *headers = NULL;
	headers = curl_slist_append(headers, "Content-Type: application/json");
	curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, 1200L);
	CURLcode res = curl_easy_perform(curl);
	curl_slist_free_all(headers);
	curl_easy_cleanup(curl);
	if (res != CURLE_OK) throw IOException("HTTP POST failed: " + std::string(curl_easy_strerror(res)));
	return response;
}

// Streaming POST: reads response line-by-line, calls callback for each line.
// Used for /commit progress streaming.
struct StreamCtx {
	std::string buffer;
	std::function<void(const std::string &)> on_line;
};

static size_t StreamWriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
	auto *ctx = static_cast<StreamCtx *>(userp);
	ctx->buffer.append((char *)contents, size * nmemb);
	// Process complete lines
	size_t pos;
	while ((pos = ctx->buffer.find('\n')) != std::string::npos) {
		std::string line = ctx->buffer.substr(0, pos);
		ctx->buffer.erase(0, pos + 1);
		if (!line.empty()) {
			ctx->on_line(line);
		}
	}
	return size * nmemb;
}

static std::string HttpPostStreaming(const std::string &url, const std::string &json_body,
                                     std::function<void(const std::string &)> on_line) {
	CURL *curl = curl_easy_init();
	if (!curl) throw IOException("Failed to init curl");
	StreamCtx ctx;
	ctx.on_line = std::move(on_line);
	struct curl_slist *headers = NULL;
	headers = curl_slist_append(headers, "Content-Type: application/json");
	curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
	curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
	curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_body.c_str());
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, StreamWriteCallback);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx);
	curl_easy_setopt(curl, CURLOPT_TIMEOUT, 1200L);
	CURLcode res = curl_easy_perform(curl);
	curl_slist_free_all(headers);
	curl_easy_cleanup(curl);
	if (res != CURLE_OK) throw IOException("HTTP POST (stream) failed: " + std::string(curl_easy_strerror(res)));
	// Process any remaining partial line
	if (!ctx.buffer.empty()) {
		ctx.on_line(ctx.buffer);
	}
	return "ok";
}

// ===========================================================================
// JSON helpers
// ===========================================================================

static std::string JsonEscape(const std::string &s) {
	std::string out;
	for (char c : s) {
		switch (c) {
		case '"': out += "\\\""; break;
		case '\\': out += "\\\\"; break;
		case '\n': out += "\\n"; break;
		case '\r': out += "\\r"; break;
		case '\t': out += "\\t"; break;
		default: out += c;
		}
	}
	return out;
}

static std::vector<std::string> JsonGetStringArray(const std::string &json, const std::string &key) {
	std::vector<std::string> result;
	std::string search = "\"" + key + "\"";
	auto pos = json.find(search);
	if (pos == std::string::npos) return result;
	pos = json.find("[", pos);
	if (pos == std::string::npos) return result;
	pos++;

	while (pos < json.size()) {
		while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == ',')) pos++;
		if (pos >= json.size() || json[pos] == ']') break;
		if (json[pos] == '"') {
			pos++;
			std::string val;
			while (pos < json.size() && json[pos] != '"') {
				if (json[pos] == '\\' && pos + 1 < json.size()) { pos++; val += json[pos]; }
				else val += json[pos];
				pos++;
			}
			pos++;
			result.push_back(val);
		} else {
			while (pos < json.size() && json[pos] != ',' && json[pos] != ']') pos++;
		}
	}
	return result;
}

static std::vector<std::vector<std::string>> JsonGetRows(const std::string &json) {
	std::vector<std::vector<std::string>> rows;
	auto pos = json.find("\"rows\"");
	if (pos == std::string::npos) return rows;
	pos = json.find("[", pos);
	if (pos == std::string::npos) return rows;
	pos++;

	while (pos < json.size()) {
		while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == ',')) pos++;
		if (pos >= json.size() || json[pos] == ']') break;

		if (json[pos] == '[') {
			pos++;
			std::vector<std::string> row;
			while (pos < json.size() && json[pos] != ']') {
				while (pos < json.size() && (json[pos] == ' ' || json[pos] == ',')) pos++;
				if (json[pos] == ']') break;
				if (json[pos] == '"') {
					pos++;
					std::string val;
					while (pos < json.size() && json[pos] != '"') {
						if (json[pos] == '\\' && pos + 1 < json.size()) { pos++; val += json[pos]; }
						else val += json[pos];
						pos++;
					}
					pos++;
					row.push_back(val);
				} else if (json[pos] == 'n' && json.substr(pos, 4) == "null") {
					row.push_back("");
					pos += 4;
				} else {
					std::string val;
					while (pos < json.size() && json[pos] != ',' && json[pos] != ']') {
						val += json[pos++];
					}
					row.push_back(val);
				}
			}
			if (pos < json.size()) pos++;
			if (!row.empty()) rows.push_back(row);
		}
	}
	return rows;
}

// Parse "columns" from /schema response: array of objects with "name", "type", "primary_key"
struct SqlLlmColumnInfo {
	std::string name;
	std::string type;
	bool primary_key = false;
};

static std::vector<SqlLlmColumnInfo> JsonGetSqlLlmColumnInfoArray(const std::string &json) {
	std::vector<SqlLlmColumnInfo> result;
	auto pos = json.find("\"columns\"");
	if (pos == std::string::npos) return result;
	pos = json.find("[", pos);
	if (pos == std::string::npos) return result;
	pos++;

	while (pos < json.size()) {
		while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == ',' || json[pos] == '\r')) pos++;
		if (pos >= json.size() || json[pos] == ']') break;

		if (json[pos] == '{') {
			// Find the matching }
			auto obj_start = pos;
			int depth = 1;
			pos++;
			while (pos < json.size() && depth > 0) {
				if (json[pos] == '{') depth++;
				else if (json[pos] == '}') depth--;
				pos++;
			}
			std::string obj = json.substr(obj_start, pos - obj_start);

			SqlLlmColumnInfo ci;
			// Extract "name"
			auto npos = obj.find("\"name\"");
			if (npos != std::string::npos) {
				npos = obj.find("\"", npos + 6);
				if (npos != std::string::npos) {
					npos++;
					while (npos < obj.size() && obj[npos] != '"') {
						ci.name += obj[npos++];
					}
				}
			}
			// Extract "type"
			auto tpos = obj.find("\"type\"");
			if (tpos != std::string::npos) {
				tpos = obj.find("\"", tpos + 6);
				if (tpos != std::string::npos) {
					tpos++;
					while (tpos < obj.size() && obj[tpos] != '"') {
						ci.type += obj[tpos++];
					}
				}
			}
			// Extract "primary_key"
			ci.primary_key = obj.find("\"primary_key\": true") != std::string::npos ||
			                 obj.find("\"primary_key\":true") != std::string::npos;

			if (!ci.name.empty()) {
				result.push_back(ci);
			}
		}
	}
	return result;
}

// Parse "tables" from /tables response
static std::vector<std::string> JsonGetTables(const std::string &json) {
	return JsonGetStringArray(json, "tables");
}

// Simple JSON string value extractor: "key": "value"
static std::string JsonGetString(const std::string &json, const std::string &key) {
	std::string search = "\"" + key + "\"";
	auto pos = json.find(search);
	if (pos == std::string::npos) return "";
	pos += search.size();
	while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':')) pos++;
	if (pos >= json.size() || json[pos] != '"') return "";
	pos++;
	std::string val;
	while (pos < json.size() && json[pos] != '"') {
		if (json[pos] == '\\' && pos + 1 < json.size()) { pos++; val += json[pos]; }
		else val += json[pos];
		pos++;
	}
	return val;
}

// Simple JSON boolean value extractor: "key": true/false
static bool JsonGetBool(const std::string &json, const std::string &key) {
	std::string search = "\"" + key + "\"";
	auto pos = json.find(search);
	if (pos == std::string::npos) return false;
	pos += search.size();
	while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':')) pos++;
	if (pos >= json.size()) return false;
	return json.substr(pos, 4) == "true";
}

// Parse array of table objects from /tables_and_schemas response:
// {"tables": [{"table": "...", "columns": [...]}, ...]}
struct SqlLlmTableWithSchema {
	std::string table_name;
	std::vector<SqlLlmColumnInfo> columns;
};

static std::vector<SqlLlmTableWithSchema> JsonGetTablesAndSchemas(const std::string &json) {
	std::vector<SqlLlmTableWithSchema> result;
	auto pos = json.find("\"tables\"");
	if (pos == std::string::npos) return result;
	pos = json.find("[", pos);
	if (pos == std::string::npos) return result;
	pos++;

	while (pos < json.size()) {
		while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == ',' || json[pos] == '\r')) pos++;
		if (pos >= json.size() || json[pos] == ']') break;

		if (json[pos] == '{') {
			// Find matching } at depth 1
			auto obj_start = pos;
			int depth = 1;
			pos++;
			while (pos < json.size() && depth > 0) {
				if (json[pos] == '{') depth++;
				else if (json[pos] == '}') depth--;
				pos++;
			}
			std::string obj = json.substr(obj_start, pos - obj_start);

			SqlLlmTableWithSchema ts;
			ts.table_name = JsonGetString(obj, "table");
			ts.columns = JsonGetSqlLlmColumnInfoArray(obj);
			if (!ts.table_name.empty()) {
				result.push_back(ts);
			}
		}
	}
	return result;
}

static LogicalType StringToLogicalType(const std::string &type_str) {
	std::string upper;
	for (char c : type_str) upper += toupper(c);
	if (upper == "INTEGER" || upper == "INT") return LogicalType::INTEGER;
	if (upper == "BIGINT") return LogicalType::BIGINT;
	if (upper == "FLOAT") return LogicalType::FLOAT;
	if (upper == "DOUBLE") return LogicalType::DOUBLE;
	if (upper == "BOOLEAN" || upper == "BOOL") return LogicalType::BOOLEAN;
	return LogicalType::VARCHAR;
}

// ===========================================================================
// SqlLlmTransaction
// ===========================================================================

SqlLlmTransaction::SqlLlmTransaction(TransactionManager &manager, ClientContext &context, string server_url_p)
    : Transaction(manager, context), server_url(std::move(server_url_p)) {
}

// ===========================================================================
// SqlLlmTransactionManager
// ===========================================================================

SqlLlmTransactionManager::SqlLlmTransactionManager(AttachedDatabase &db, string server_url_p)
    : TransactionManager(db), server_url(std::move(server_url_p)) {
}

Transaction &SqlLlmTransactionManager::StartTransaction(ClientContext &context) {
	lock_guard<mutex> guard(transaction_lock);
	auto txn = make_uniq<SqlLlmTransaction>(*this, context, server_url);
	auto &ref = *txn;
	transactions.push_back(std::move(txn));
	return ref;
}

ErrorData SqlLlmTransactionManager::CommitTransaction(ClientContext &context, Transaction &transaction) {
	try {
		// Stream progress from server
		HttpPostStreaming(server_url + "/commit", "{}", [](const std::string &line) {
			// Parse progress and print to stderr
			auto status = JsonGetString(line, "status");
			if (status == "training") {
				auto epoch_pos = line.find("\"epoch\"");
				auto total_pos = line.find("\"total_epochs\"");
				auto loss_pos = line.find("\"loss\"");
				auto pct_pos = line.find("\"pct\"");

				// Simple number extraction after key
				auto extract_num = [&](size_t kpos) -> std::string {
					if (kpos == std::string::npos) return "?";
					auto colon = line.find(':', kpos);
					if (colon == std::string::npos) return "?";
					colon++;
					while (colon < line.size() && line[colon] == ' ') colon++;
					std::string num;
					while (colon < line.size() && (isdigit(line[colon]) || line[colon] == '.')) {
						num += line[colon++];
					}
					return num.empty() ? "?" : num;
				};

				auto epoch_val = extract_num(epoch_pos);
				auto total_val = extract_num(total_pos);
				auto loss_val = extract_num(loss_pos);
				auto pct_val = extract_num(pct_pos);

				fprintf(stderr, "\rFine-tuning: epoch %s/%s (%s%%) loss=%s   ",
				        epoch_val.c_str(), total_val.c_str(), pct_val.c_str(), loss_val.c_str());
				
			} else if (status == "done") {
				fprintf(stderr, "\rFine-tuning: complete!                              \n");
				
			} else if (status == "nothing_to_commit") {
				// No pending changes, nothing to do
			}
		});
		return ErrorData();
	} catch (std::exception &e) {
		return ErrorData(e);
	}
}

void SqlLlmTransactionManager::RollbackTransaction(Transaction &transaction) {
	try {
		HttpPost(server_url + "/rollback", "{}");
	} catch (...) {
		// Best effort
	}
}

void SqlLlmTransactionManager::Checkpoint(ClientContext &context, bool force) {
	// No-op
}

unique_ptr<TransactionManager> SqlLlmTransactionManager::Create(optional_ptr<StorageExtensionInfo> storage_info,
                                                                 AttachedDatabase &db, Catalog &catalog) {
	auto &llm_catalog = catalog.Cast<SqlLlmCatalog>();
	return make_uniq<SqlLlmTransactionManager>(db, llm_catalog.server_url);
}

// ===========================================================================
// SqlLlmTableEntry — scan function data
// ===========================================================================

struct SqlLlmScanBindData : public TableFunctionData {
	string server_url;
	string table_name;
	vector<string> column_names;
	vector<LogicalType> column_types;
};

struct SqlLlmScanState : public GlobalTableFunctionState {
	bool fetched = false;
	idx_t current_row = 0;
	std::vector<std::vector<std::string>> rows;  // use std:: explicitly for JSON helper compatibility
	vector<column_t> projected_columns;           // which bind_data columns are needed (projection pushdown)
};

static unique_ptr<GlobalTableFunctionState> SqlLlmScanInit(ClientContext &context,
                                                            TableFunctionInitInput &input) {
	auto state = make_uniq<SqlLlmScanState>();
	// Store projected column indices for use in the scan function
	state->projected_columns = input.column_ids;
	return std::move(state);
}

static void SqlLlmScanFunc(ClientContext &context, TableFunctionInput &input, DataChunk &output) {
	auto &bind_data = input.bind_data->Cast<SqlLlmScanBindData>();
	auto &state = input.global_state->Cast<SqlLlmScanState>();

	auto &column_ids = state.projected_columns;

	if (!state.fetched) {
		state.fetched = true;

		// Only request the projected columns from the server
		std::vector<std::string> projected_names;
		for (auto &col_id : column_ids) {
			if (col_id < bind_data.column_names.size()) {
				projected_names.push_back(bind_data.column_names[col_id]);
			}
		}

		std::string body = "{\"table\": \"" + JsonEscape(bind_data.table_name) + "\", \"columns\": [";
		for (idx_t i = 0; i < projected_names.size(); i++) {
			if (i > 0) body += ", ";
			body += "\"" + JsonEscape(projected_names[i]) + "\"";
		}
		body += "]}";

		{
			ProgressSpinner spinner("Querying '" + bind_data.table_name + "' via LLM...");
			std::string response = HttpPost(bind_data.server_url + "/query", body);
			auto parsed = JsonGetRows(response);
			for (auto &r : parsed) {
				state.rows.push_back(r);
			}
		}
	}

	if (state.current_row >= state.rows.size()) {
		output.SetCardinality(0);
		return;
	}

	idx_t count = 0;
	idx_t num_output_cols = column_ids.size();

	while (state.current_row < state.rows.size() && count < STANDARD_VECTOR_SIZE) {
		auto &row = state.rows[state.current_row];
		for (idx_t out_col = 0; out_col < num_output_cols; out_col++) {
			auto bind_col = column_ids[out_col];
			// row[out_col] corresponds to the out_col-th projected column
			if (out_col < row.size() && !row[out_col].empty()) {
				auto &target_type = bind_data.column_types[bind_col];
				if (target_type == LogicalType::VARCHAR) {
					output.data[out_col].SetValue(count, Value(row[out_col]));
				} else {
					Value val(row[out_col]);
					if (val.DefaultTryCastAs(target_type)) {
						output.data[out_col].SetValue(count, val);
					} else {
						output.data[out_col].SetValue(count, Value(target_type));
					}
				}
			} else {
				output.data[out_col].SetValue(count, Value(bind_data.column_types[bind_col]));
			}
		}
		state.current_row++;
		count++;
	}

	output.SetCardinality(count);
}

// ===========================================================================
// SqlLlmTableEntry
// ===========================================================================

SqlLlmTableEntry::SqlLlmTableEntry(Catalog &catalog, SchemaCatalogEntry &schema,
                                     CreateTableInfo &info, string server_url_p)
    : TableCatalogEntry(catalog, schema, info), server_url(std::move(server_url_p)) {
}

TableFunction SqlLlmTableEntry::GetScanFunction(ClientContext &context, unique_ptr<FunctionData> &bind_data) {
	auto data = make_uniq<SqlLlmScanBindData>();
	data->server_url = server_url;
	data->table_name = name;

	for (auto &col : columns.Logical()) {
		data->column_names.push_back(col.Name());
		data->column_types.push_back(col.Type());
	}

	bind_data = std::move(data);

	TableFunction func("sql_llm_catalog_scan", {}, SqlLlmScanFunc);
	func.init_global = SqlLlmScanInit;
	func.projection_pushdown = true;
	return func;
}

unique_ptr<BaseStatistics> SqlLlmTableEntry::GetStatistics(ClientContext &context, column_t column_id) {
	return nullptr;
}

TableStorageInfo SqlLlmTableEntry::GetStorageInfo(ClientContext &context) {
	return TableStorageInfo();
}

// ===========================================================================
// SqlLlmSchemaEntry
// ===========================================================================

SqlLlmSchemaEntry::SqlLlmSchemaEntry(Catalog &catalog, CreateSchemaInfo &info, string server_url_p)
    : SchemaCatalogEntry(catalog, info), server_url(std::move(server_url_p)) {
}

optional_ptr<CatalogEntry> SqlLlmSchemaEntry::CreateTable(CatalogTransaction transaction,
                                                           BoundCreateTableInfo &info) {
	auto &create_info = info.Base();

	// Build structured JSON for server
	std::string body = "{\"table\": \"" + JsonEscape(create_info.table) + "\", \"columns\": [";
	bool first = true;
	for (auto &col : create_info.columns.Logical()) {
		if (!first) body += ", ";
		first = false;
		body += "{\"name\": \"" + JsonEscape(col.Name()) + "\", \"type\": \"" +
		        JsonEscape(col.Type().ToString()) + "\"";
		// Check if primary key (simple heuristic: check constraints)
		bool is_pk = false;
		for (auto &constraint : create_info.constraints) {
			if (constraint->type == ConstraintType::UNIQUE) {
				// Primary key constraints are UNIQUE constraints with is_primary_key flag
				// We check if this column is in the constraint
				// For simplicity, mark if name matches
			}
		}
		body += ", \"primary_key\": " + std::string(is_pk ? "true" : "false") + "}";
	}
	body += "]}";

	HttpPost(server_url + "/create_table", body);

	auto entry = make_uniq<SqlLlmTableEntry>(catalog, *this, create_info, server_url);
	auto *ptr = entry.get();
	owned_entries.push_back(std::move(entry));
	return ptr;
}

optional_ptr<CatalogEntry> SqlLlmSchemaEntry::LookupEntry(CatalogTransaction transaction,
                                                           const EntryLookupInfo &lookup_info) {
	if (lookup_info.GetCatalogType() != CatalogType::TABLE_ENTRY) {
		return nullptr;
	}

	auto table_name = lookup_info.GetEntryName();

	// Check locally-created entries first (from CreateTable in this session)
	for (auto &entry : owned_entries) {
		if (entry->name == table_name) {
			return entry.get();
		}
	}

	// Query LLM via server — combined lookup: existence check + schema in one HTTP call
	try {
		auto spinner = make_uniq<ProgressSpinner>("Looking up table '" + table_name + "'...");
		std::string response = HttpGet(server_url + "/lookup/" + table_name);

		// Check if table exists
		if (!JsonGetBool(response, "exists")) {
			spinner.reset();
			return nullptr;
		}

		auto columns_info = JsonGetSqlLlmColumnInfoArray(response);
		if (columns_info.empty()) {
			return nullptr;
		}

		auto create_info = make_uniq<CreateTableInfo>();
		create_info->catalog = catalog.GetName();
		create_info->schema = name;  // schema name
		create_info->table = table_name;
		std::set<std::string> seen_cols;
		for (auto &ci : columns_info) {
			if (seen_cols.count(ci.name)) continue;
			seen_cols.insert(ci.name);
			if (ci.name.empty()) continue;
			bool valid = true;
			for (char ch : ci.name) {
				if (!isalnum(ch) && ch != '_') { valid = false; break; }
			}
			if (!valid) continue;
			if (isdigit(ci.name[0])) continue;
			create_info->columns.AddColumn(ColumnDefinition(ci.name, StringToLogicalType(ci.type)));
		}

		if (create_info->columns.LogicalColumnCount() == 0) {
			return nullptr;
		}

		create_info->columns.Finalize();
		auto entry = make_uniq<SqlLlmTableEntry>(catalog, *this, *create_info, server_url);
		auto *ptr = entry.get();
		owned_entries.push_back(std::move(entry));
		return ptr;
	} catch (...) {
		return nullptr;
	}
}

void SqlLlmSchemaEntry::Scan(ClientContext &context, CatalogType type,
                              const std::function<void(CatalogEntry &)> &callback) {
	if (type != CatalogType::TABLE_ENTRY) return;

	try {
		auto spinner = make_uniq<ProgressSpinner>("Querying LLM for tables and schemas...");
		// Single HTTP call to get all tables with their schemas
		std::string response = HttpGet(server_url + "/tables_and_schemas");
		auto tables_with_schemas = JsonGetTablesAndSchemas(response);

		for (size_t i = 0; i < tables_with_schemas.size(); i++) {
			auto &ts = tables_with_schemas[i];
			spinner->update("Processing: " + ts.table_name + " (" + std::to_string(i + 1) + "/" + std::to_string(tables_with_schemas.size()) + ")");
			if (ts.columns.empty()) continue;

			auto create_info = make_uniq<CreateTableInfo>();
			create_info->table = ts.table_name;
			for (auto &ci : ts.columns) {
				create_info->columns.AddColumn(ColumnDefinition(ci.name, StringToLogicalType(ci.type)));
			}

			auto entry = make_uniq<SqlLlmTableEntry>(catalog, *this, *create_info, server_url);
			callback(*entry);
			owned_entries.push_back(std::move(entry));
		}
		spinner.reset(); // stop spinner, clear line
	} catch (...) {
		// If server unreachable, return empty
	}
}

void SqlLlmSchemaEntry::Scan(CatalogType type, const std::function<void(CatalogEntry &)> &callback) {
	// No committed-only scan variant needed — delegate to main scan
	// We can't do LLM inference without a ClientContext, so just return empty
}

void SqlLlmSchemaEntry::DropEntry(ClientContext &context, DropInfo &info) {
	// No-op — LLM will forget after retraining without data
}

void SqlLlmSchemaEntry::Alter(CatalogTransaction transaction, AlterInfo &info) {
	throw NotImplementedException("ALTER not supported for SQL-LLM tables");
}

optional_ptr<CatalogEntry> SqlLlmSchemaEntry::CreateFunction(CatalogTransaction transaction,
                                                              CreateFunctionInfo &info) {
	throw NotImplementedException("CREATE FUNCTION not supported in SQL-LLM catalog");
}
optional_ptr<CatalogEntry> SqlLlmSchemaEntry::CreateView(CatalogTransaction transaction, CreateViewInfo &info) {
	throw NotImplementedException("CREATE VIEW not supported in SQL-LLM catalog");
}
optional_ptr<CatalogEntry> SqlLlmSchemaEntry::CreateSequence(CatalogTransaction transaction,
                                                              CreateSequenceInfo &info) {
	throw NotImplementedException("CREATE SEQUENCE not supported in SQL-LLM catalog");
}
optional_ptr<CatalogEntry> SqlLlmSchemaEntry::CreateTableFunction(CatalogTransaction transaction,
                                                                    CreateTableFunctionInfo &info) {
	throw NotImplementedException("CREATE TABLE FUNCTION not supported in SQL-LLM catalog");
}
optional_ptr<CatalogEntry> SqlLlmSchemaEntry::CreateCopyFunction(CatalogTransaction transaction,
                                                                   CreateCopyFunctionInfo &info) {
	throw NotImplementedException("CREATE COPY FUNCTION not supported in SQL-LLM catalog");
}
optional_ptr<CatalogEntry> SqlLlmSchemaEntry::CreatePragmaFunction(CatalogTransaction transaction,
                                                                     CreatePragmaFunctionInfo &info) {
	throw NotImplementedException("CREATE PRAGMA FUNCTION not supported in SQL-LLM catalog");
}
optional_ptr<CatalogEntry> SqlLlmSchemaEntry::CreateCollation(CatalogTransaction transaction,
                                                               CreateCollationInfo &info) {
	throw NotImplementedException("CREATE COLLATION not supported in SQL-LLM catalog");
}
optional_ptr<CatalogEntry> SqlLlmSchemaEntry::CreateType(CatalogTransaction transaction, CreateTypeInfo &info) {
	throw NotImplementedException("CREATE TYPE not supported in SQL-LLM catalog");
}
optional_ptr<CatalogEntry> SqlLlmSchemaEntry::CreateIndex(CatalogTransaction transaction, CreateIndexInfo &info,
                                                           TableCatalogEntry &table) {
	throw NotImplementedException("CREATE INDEX not supported in SQL-LLM catalog");
}

// ===========================================================================
// SqlLlmCatalog
// ===========================================================================

SqlLlmCatalog::SqlLlmCatalog(AttachedDatabase &db, string server_url_p)
    : Catalog(db), server_url(std::move(server_url_p)) {
}

void SqlLlmCatalog::Initialize(bool load_builtin) {
	CreateSchemaInfo info;
	info.schema = "main";
	info.on_conflict = OnCreateConflict::IGNORE_ON_CONFLICT;
	default_schema = make_uniq<SqlLlmSchemaEntry>(*this, info, server_url);
}

optional_ptr<CatalogEntry> SqlLlmCatalog::CreateSchema(CatalogTransaction transaction, CreateSchemaInfo &info) {
	if (info.schema == "main" || info.schema == DEFAULT_SCHEMA) {
		return default_schema.get();
	}
	throw NotImplementedException("SQL-LLM only supports the 'main' schema");
}

optional_ptr<SchemaCatalogEntry> SqlLlmCatalog::LookupSchema(CatalogTransaction transaction,
                                                              const EntryLookupInfo &schema_lookup,
                                                              OnEntryNotFound if_not_found) {
	auto schema_name = schema_lookup.GetEntryName();
	if (schema_name == "main" || schema_name == DEFAULT_SCHEMA || schema_name.empty()) {
		return default_schema.get();
	}
	if (if_not_found == OnEntryNotFound::RETURN_NULL) {
		return nullptr;
	}
	throw CatalogException("Schema '%s' not found in SQL-LLM catalog", schema_name);
}

void SqlLlmCatalog::ScanSchemas(ClientContext &context, std::function<void(SchemaCatalogEntry &)> callback) {
	callback(*default_schema);
}

PhysicalOperator &SqlLlmCatalog::PlanCreateTableAs(ClientContext &context, PhysicalPlanGenerator &planner,
                                                     LogicalCreateTable &op, PhysicalOperator &plan) {
	// CREATE TABLE AS SELECT — not really meaningful for LLM, but implement for completeness
	// Just create the table and ignore the SELECT data (the LLM can't store arbitrary query results)
	auto &create = planner.Make<PhysicalCreateTable>(op, op.schema, std::move(op.info), op.estimated_cardinality);
	return create;
}

// ---------------------------------------------------------------------------
// PhysicalSqlLlmInsert — custom sink for INSERT
// ---------------------------------------------------------------------------

struct SqlLlmInsertGlobalState : public GlobalSinkState {
	mutex lock;
	vector<vector<string>> rows;
	idx_t insert_count = 0;
};

struct SqlLlmInsertSourceState : public GlobalSourceState {
	bool returned = false;
};

class PhysicalSqlLlmInsert : public PhysicalOperator {
public:
	PhysicalSqlLlmInsert(PhysicalPlan &physical_plan, vector<LogicalType> types,
	                      string server_url, string table_name,
	                      vector<string> column_names, vector<LogicalType> column_types,
	                      idx_t estimated_cardinality)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, std::move(types), estimated_cardinality),
	      server_url(std::move(server_url)), table_name(std::move(table_name)),
	      column_names(std::move(column_names)), column_types(std::move(column_types)) {
	}

	string server_url;
	string table_name;
	vector<string> column_names;
	vector<LogicalType> column_types;

	bool IsSink() const override { return true; }
	bool IsSource() const override { return true; }
	bool ParallelSink() const override { return false; }

	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override {
		return make_uniq<SqlLlmInsertGlobalState>();
	}

	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override {
		auto &gstate = input.global_state.Cast<SqlLlmInsertGlobalState>();
		lock_guard<mutex> guard(gstate.lock);

		for (idx_t r = 0; r < chunk.size(); r++) {
			vector<string> row;
			for (idx_t c = 0; c < chunk.ColumnCount(); c++) {
				auto val = chunk.data[c].GetValue(r);
				if (val.IsNull()) {
					row.push_back("");
				} else {
					row.push_back(val.ToString());
				}
			}
			gstate.rows.push_back(std::move(row));
		}

		return SinkResultType::NEED_MORE_INPUT;
	}

	SinkFinalizeType Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
	                           OperatorSinkFinalizeInput &input) const override {
		auto &gstate = input.global_state.Cast<SqlLlmInsertGlobalState>();

		if (gstate.rows.empty()) {
			return SinkFinalizeType::READY;
		}

		// Build structured JSON
		std::string body = "{\"table\": \"" + JsonEscape(table_name) + "\", \"columns\": [";
		for (idx_t i = 0; i < column_names.size(); i++) {
			if (i > 0) body += ", ";
			body += "\"" + JsonEscape(column_names[i]) + "\"";
		}
		body += "], \"rows\": [";

		for (idx_t r = 0; r < gstate.rows.size(); r++) {
			if (r > 0) body += ", ";
			body += "[";
			auto &row = gstate.rows[r];
			for (idx_t c = 0; c < row.size(); c++) {
				if (c > 0) body += ", ";
				body += "\"" + JsonEscape(row[c]) + "\"";
			}
			body += "]";
		}
		body += "]}";

		HttpPost(server_url + "/insert", body);
		gstate.insert_count = gstate.rows.size();

		return SinkFinalizeType::READY;
	}

	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &context) const override {
		return make_uniq<SqlLlmInsertSourceState>();
	}

	SourceResultType GetData(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const override {
		auto &source_state = input.global_state.Cast<SqlLlmInsertSourceState>();
		if (source_state.returned) {
			return SourceResultType::FINISHED;
		}
		source_state.returned = true;

		auto &gstate = sink_state->Cast<SqlLlmInsertGlobalState>();
		chunk.SetCardinality(1);
		chunk.SetValue(0, 0, Value::BIGINT(gstate.insert_count));
		return SourceResultType::HAVE_MORE_OUTPUT;
	}
};

PhysicalOperator &SqlLlmCatalog::PlanInsert(ClientContext &context, PhysicalPlanGenerator &planner,
                                              LogicalInsert &op, optional_ptr<PhysicalOperator> plan) {
	// Extract table info
	auto &table = op.table.Cast<SqlLlmTableEntry>();
	vector<string> col_names;
	vector<LogicalType> col_types;
	for (auto &col : table.GetColumns().Physical()) {
		col_names.push_back(col.Name());
		col_types.push_back(col.Type());
	}

	// Handle column_index_map (partial column list) — resolve defaults
	if (!op.column_index_map.empty()) {
		plan = planner.ResolveDefaultsProjection(op, *plan);
	}

	auto &insert = planner.Make<PhysicalSqlLlmInsert>(
	    op.types, server_url, table.name, std::move(col_names), std::move(col_types),
	    op.estimated_cardinality);

	if (plan) {
		insert.children.push_back(*plan);
	}
	return insert;
}

PhysicalOperator &SqlLlmCatalog::PlanDelete(ClientContext &context, PhysicalPlanGenerator &planner,
                                              LogicalDelete &op, PhysicalOperator &plan) {
	throw NotImplementedException("DELETE not supported for SQL-LLM tables");
}

PhysicalOperator &SqlLlmCatalog::PlanUpdate(ClientContext &context, PhysicalPlanGenerator &planner,
                                              LogicalUpdate &op, PhysicalOperator &plan) {
	throw NotImplementedException("UPDATE not supported for SQL-LLM tables");
}

DatabaseSize SqlLlmCatalog::GetDatabaseSize(ClientContext &context) {
	return DatabaseSize();
}

void SqlLlmCatalog::DropSchema(ClientContext &context, DropInfo &info) {
	// No-op
}

unique_ptr<Catalog> SqlLlmCatalog::Attach(optional_ptr<StorageExtensionInfo> storage_info,
                                           ClientContext &context, AttachedDatabase &db,
                                           const string &name, AttachInfo &info, AttachOptions &options) {
	string server_url = "http://localhost:8000";

	// Check for server_url in attach options
	if (info.options.count("server_url")) {
		server_url = info.options["server_url"].ToString();
	}
	// Also check path — if non-empty, treat as server URL
	if (!info.path.empty()) {
		server_url = info.path;
	}

	auto catalog = make_uniq<SqlLlmCatalog>(db, server_url);
	catalog->Initialize(false);
	return std::move(catalog);
}

// ===========================================================================
// Extension registration
// ===========================================================================

static void LoadInternal(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();

	// Register storage extension for ATTACH ... (TYPE SQL_LLM)
	auto storage_ext = make_uniq<StorageExtension>();
	storage_ext->storage_info = make_shared_ptr<SqlLlmStorageInfo>("http://localhost:8000");
	storage_ext->attach = SqlLlmCatalog::Attach;
	storage_ext->create_transaction_manager = SqlLlmTransactionManager::Create;
	db.config.storage_extensions["sql_llm"] = std::move(storage_ext);

	// Keep old scalar functions for backward compatibility
	loader.RegisterFunction(ScalarFunction("sql_llm_set_server", {LogicalType::VARCHAR},
	                                        LogicalType::VARCHAR, [](DataChunk &args, ExpressionState &state, Vector &result) {
		// No-op now — server URL is set via ATTACH path
		UnaryExecutor::Execute<string_t, string_t>(args.data[0], result, args.size(), [&](string_t url) {
			return StringVector::AddString(result, "Use ATTACH '<url>' AS name (TYPE SQL_LLM) instead");
		});
	}));
}

void SqlLlmExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}
std::string SqlLlmExtension::Name() { return "sql_llm"; }
std::string SqlLlmExtension::Version() const { return "0.3.0"; }

} // namespace duckdb

extern "C" {
DUCKDB_CPP_EXTENSION_ENTRY(sql_llm, loader) {
	duckdb::LoadInternal(loader);
}
}
