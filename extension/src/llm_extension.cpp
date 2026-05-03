#define DUCKDB_EXTENSION_MAIN

#include "llm_extension.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/constants.hpp"
#include "duckdb/common/enums/expression_type.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/common/http_util.hpp"
#include "duckdb/common/reference_map.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/client_context_state.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/main/prepared_statement_data.hpp"
#include "duckdb/optimizer/optimizer_extension.hpp"
#include "duckdb/parser/constraints/not_null_constraint.hpp"
#include "duckdb/parser/constraints/unique_constraint.hpp"
#include "duckdb/parser/parsed_data/alter_info.hpp"
#include "duckdb/parser/parsed_data/attach_info.hpp"
#include "duckdb/parser/parsed_data/create_collation_info.hpp"
#include "duckdb/parser/parsed_data/create_copy_function_info.hpp"
#include "duckdb/parser/parsed_data/create_function_info.hpp"
#include "duckdb/parser/parsed_data/create_index_info.hpp"
#include "duckdb/parser/parsed_data/create_pragma_function_info.hpp"
#include "duckdb/parser/parsed_data/create_schema_info.hpp"
#include "duckdb/parser/parsed_data/create_sequence_info.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/parser/parsed_data/create_type_info.hpp"
#include "duckdb/parser/parsed_data/create_view_info.hpp"
#include "duckdb/parser/parsed_data/drop_info.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/operator/logical_create_table.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_update.hpp"
#include "duckdb/planner/parsed_data/bound_create_table_info.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/database_size.hpp"
#include "duckdb/storage/storage_extension.hpp"
#include "duckdb/storage/table_storage_info.hpp"
#include "duckdb/transaction/transaction.hpp"
#include "duckdb/transaction/transaction_manager.hpp"

#include "httplib.hpp"
#include "yyjson.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>

namespace duckdb {
using namespace duckdb_yyjson; // NOLINT

class LlmCatalog;
class LlmSchemaEntry;
class LlmTableEntry;

class JsonWriteDocument {
public:
	JsonWriteDocument() : doc(yyjson_mut_doc_new(nullptr)) {
		if (!doc) {
			throw InternalException("Failed to allocate LLM adapter JSON document");
		}
	}
	JsonWriteDocument(const JsonWriteDocument &) = delete;
	JsonWriteDocument &operator=(const JsonWriteDocument &) = delete;
	~JsonWriteDocument() {
		if (doc) {
			yyjson_mut_doc_free(doc);
		}
	}
	yyjson_mut_doc *Doc() {
		return doc;
	}
	yyjson_mut_val *Object() {
		return yyjson_mut_obj(doc);
	}
	yyjson_mut_val *Array() {
		return yyjson_mut_arr(doc);
	}
	yyjson_mut_val *String(const string &value) {
		return yyjson_mut_strncpy(doc, value.c_str(), value.size());
	}
	yyjson_mut_val *String(const char *value) {
		return yyjson_mut_strcpy(doc, value);
	}
	yyjson_mut_val *Bool(bool value) {
		return yyjson_mut_bool(doc, value);
	}
	yyjson_mut_val *Int(int64_t value) {
		return yyjson_mut_int(doc, value);
	}
	yyjson_mut_val *Double(double value) {
		return yyjson_mut_real(doc, value);
	}
	yyjson_mut_val *Null() {
		return yyjson_mut_null(doc);
	}
	string Write(yyjson_mut_val *root) {
		yyjson_mut_doc_set_root(doc, root);
		size_t len = 0;
		yyjson_write_err error;
		auto data = yyjson_mut_write_opts(doc, YYJSON_WRITE_NOFLAG, nullptr, &len, &error);
		if (!data) {
			throw IOException("Failed to serialize LLM adapter JSON: %s", error.msg ? error.msg : "unknown error");
		}
		string result(data, len);
		std::free(data);
		return result;
	}
	yyjson_mut_val *ParseAndCopy(const string &json, const string &context) {
		string input = json;
		yyjson_read_err error;
		auto parsed = yyjson_read_opts(input.empty() ? nullptr : &input[0], input.size(), YYJSON_READ_NOFLAG, nullptr,
		                               &error);
		if (!parsed) {
			throw InvalidInputException("Malformed LLM adapter JSON in %s at byte %llu: %s", context, error.pos,
			                            error.msg ? error.msg : "unknown error");
		}
		auto root = yyjson_doc_get_root(parsed);
		auto copied = yyjson_val_mut_copy(doc, root);
		yyjson_doc_free(parsed);
		if (!copied) {
			throw InternalException("Failed to copy LLM adapter JSON value for %s", context);
		}
		return copied;
	}

private:
	yyjson_mut_doc *doc;
};

class JsonReadDocument {
public:
	explicit JsonReadDocument(string input_p, string context_p) : input(std::move(input_p)), context(std::move(context_p)) {
		yyjson_read_err error;
		doc = yyjson_read_opts(input.empty() ? nullptr : &input[0], input.size(), YYJSON_READ_NOFLAG, nullptr, &error);
		if (!doc) {
			throw InvalidInputException("Malformed LLM adapter JSON in %s at byte %llu: %s", context, error.pos,
			                            error.msg ? error.msg : "unknown error");
		}
	}
	JsonReadDocument(const JsonReadDocument &) = delete;
	JsonReadDocument &operator=(const JsonReadDocument &) = delete;
	~JsonReadDocument() {
		if (doc) {
			yyjson_doc_free(doc);
		}
	}
	yyjson_val *Root() const {
		auto root = yyjson_doc_get_root(doc);
		if (!root) {
			throw InvalidInputException("Malformed LLM adapter JSON in %s: missing root value", context);
		}
		return root;
	}

private:
	string input;
	string context;
	yyjson_doc *doc;
};

static void JsonAdd(JsonWriteDocument &doc, yyjson_mut_val *object, const char *key, yyjson_mut_val *value) {
	if (!yyjson_mut_obj_add_val(doc.Doc(), object, key, value)) {
		throw InternalException("Failed to add JSON field \"%s\"", key);
	}
}

static void JsonAppend(yyjson_mut_val *array, yyjson_mut_val *value) {
	if (!yyjson_mut_arr_add_val(array, value)) {
		throw InternalException("Failed to append JSON array value");
	}
}

static yyjson_val *JsonRequire(yyjson_val *object, const char *key, const string &context) {
	if (!yyjson_is_obj(object)) {
		throw InvalidInputException("Malformed LLM adapter JSON: expected object for %s", context);
	}
	auto value = yyjson_obj_get(object, key);
	if (!value) {
		throw InvalidInputException("Malformed LLM adapter JSON: missing field \"%s\" in %s", key, context);
	}
	return value;
}

static yyjson_val *JsonGet(yyjson_val *object, const char *key) {
	if (!yyjson_is_obj(object)) {
		return nullptr;
	}
	return yyjson_obj_get(object, key);
}

static yyjson_val *JsonArray(yyjson_val *value, const string &context) {
	if (!yyjson_is_arr(value)) {
		throw InvalidInputException("Malformed LLM adapter JSON: expected array for %s", context);
	}
	return value;
}

static string JsonString(yyjson_val *value, const string &context) {
	if (!yyjson_is_str(value)) {
		throw InvalidInputException("Malformed LLM adapter JSON: expected string for %s", context);
	}
	return string(yyjson_get_str(value), yyjson_get_len(value));
}

static bool JsonBoolean(yyjson_val *value, const string &context) {
	if (yyjson_is_true(value)) {
		return true;
	}
	if (yyjson_is_false(value)) {
		return false;
	}
	throw InvalidInputException("Malformed LLM adapter JSON: expected boolean for %s", context);
}

static int64_t JsonInteger(yyjson_val *value, const string &context) {
	if (!yyjson_is_int(value)) {
		throw InvalidInputException("Malformed LLM adapter JSON: expected integer for %s", context);
	}
	return yyjson_get_sint(value);
}

static double JsonNumber(yyjson_val *value, const string &context) {
	if (yyjson_is_int(value)) {
		return static_cast<double>(yyjson_get_sint(value));
	}
	if (yyjson_is_real(value)) {
		return yyjson_get_real(value);
	}
	throw InvalidInputException("Malformed LLM adapter JSON: expected number for %s", context);
}

struct LlmColumnMeta {
	string name;
	LogicalType type;
	bool nullable;
};

struct LlmTableMeta {
	string schema;
	string name;
	vector<LlmColumnMeta> columns;
	vector<string> primary_key;
};

struct LlmCatalogSnapshot {
	string version;
	vector<LlmTableMeta> tables;
};

static string NormalizeEndpoint(string endpoint) {
	if (endpoint.empty()) {
		endpoint = "http://0.0.0.0:5366";
	}
	while (!endpoint.empty() && endpoint.back() == '/') {
		endpoint.pop_back();
	}
	return endpoint;
}

struct ParsedHttpEndpoint {
	string proto_host_port;
	string path_prefix;
};

static ParsedHttpEndpoint ParseHttpEndpoint(const string &endpoint) {
	auto endpoint_with_path = endpoint;
	if (endpoint_with_path.find('/', strlen("http://")) == string::npos &&
	    endpoint_with_path.find('/', strlen("https://")) == string::npos) {
		endpoint_with_path += "/";
	}
	ParsedHttpEndpoint result;
	HTTPUtil::DecomposeURL(endpoint_with_path, result.path_prefix, result.proto_host_port);
	if (result.path_prefix == "/") {
		result.path_prefix.clear();
	}
	return result;
}

static string HttpPostJson(const ParsedHttpEndpoint &endpoint, const string &path, const string &body) {
	auto full_path = endpoint.path_prefix.empty() ? path : endpoint.path_prefix + path;
	duckdb_httplib::Client client(endpoint.proto_host_port);
	client.set_keep_alive(false);
	client.set_decompress(false);
	duckdb_httplib::Headers headers = {
	    {"Accept", "application/json"},
	};
	auto response = client.Post(full_path, headers, body, "application/json");
	if (!response) {
		throw IOException("LLM adapter request to %s%s failed: %s", endpoint.proto_host_port, full_path,
		                  to_string(response.error()));
	}
	if (response->status < 200 || response->status >= 300) {
		throw IOException("LLM adapter request to %s%s failed with HTTP %d: %s", endpoint.proto_host_port, full_path,
		                  response->status, response->body);
	}
	return response->body;
}

static string GetAttachStringOption(AttachInfo &info, const string &key, const string &default_value) {
	auto entry = info.options.find(key);
	if (entry == info.options.end() || entry->second.IsNull()) {
		return default_value;
	}
	return entry->second.GetValue<string>();
}

static bool IsSupportedAdapterType(const LogicalType &type) {
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
	case LogicalTypeId::TINYINT:
	case LogicalTypeId::SMALLINT:
	case LogicalTypeId::INTEGER:
	case LogicalTypeId::BIGINT:
	case LogicalTypeId::FLOAT:
	case LogicalTypeId::DOUBLE:
	case LogicalTypeId::VARCHAR:
		return true;
	default:
		return false;
	}
}

static LogicalType ParseAdapterType(ClientContext &context, const string &type_string) {
	auto type = TransformStringToLogicalType(type_string, context);
	if (!IsSupportedAdapterType(type)) {
		throw NotImplementedException("LLM adapter does not support DuckDB type \"%s\" yet", type.ToString());
	}
	return type;
}

static yyjson_mut_val *ValueToJson(JsonWriteDocument &doc, const Value &value) {
	if (value.IsNull()) {
		return doc.Null();
	}
	switch (value.type().id()) {
	case LogicalTypeId::BOOLEAN:
		return doc.Bool(value.GetValue<bool>());
	case LogicalTypeId::TINYINT:
		return doc.Int(static_cast<int64_t>(value.GetValue<int8_t>()));
	case LogicalTypeId::SMALLINT:
		return doc.Int(static_cast<int64_t>(value.GetValue<int16_t>()));
	case LogicalTypeId::INTEGER:
		return doc.Int(static_cast<int64_t>(value.GetValue<int32_t>()));
	case LogicalTypeId::BIGINT:
		return doc.Int(value.GetValue<int64_t>());
	case LogicalTypeId::FLOAT:
		return doc.Double(static_cast<double>(value.GetValue<float>()));
	case LogicalTypeId::DOUBLE:
		return doc.Double(value.GetValue<double>());
	case LogicalTypeId::VARCHAR:
		return doc.String(value.GetValue<string>());
	default:
		throw NotImplementedException("LLM adapter cannot serialize literal of type \"%s\" yet", value.type().ToString());
	}
}

static Value JsonToValue(yyjson_val *json, const LogicalType &type, const string &context) {
	if (yyjson_is_null(json)) {
		return Value(type);
	}
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
		return Value::BOOLEAN(JsonBoolean(json, context));
	case LogicalTypeId::TINYINT:
		return Value::TINYINT(NumericCast<int8_t>(JsonInteger(json, context)));
	case LogicalTypeId::SMALLINT:
		return Value::SMALLINT(NumericCast<int16_t>(JsonInteger(json, context)));
	case LogicalTypeId::INTEGER:
		return Value::INTEGER(NumericCast<int32_t>(JsonInteger(json, context)));
	case LogicalTypeId::BIGINT:
		return Value::BIGINT(JsonInteger(json, context));
	case LogicalTypeId::FLOAT:
		return Value(static_cast<float>(JsonNumber(json, context)));
	case LogicalTypeId::DOUBLE:
		return Value::DOUBLE(JsonNumber(json, context));
	case LogicalTypeId::VARCHAR:
		return Value(JsonString(json, context));
	default:
		throw NotImplementedException("LLM adapter cannot materialize values of type \"%s\" yet", type.ToString());
	}
}

static yyjson_mut_val *ColumnReferenceToJson(JsonWriteDocument &doc, const LogicalGet &get,
                                             const BoundColumnRefExpression &column) {
	auto binding_index = column.binding.column_index;
	auto &column_ids = get.GetColumnIds();
	if (binding_index >= column_ids.size()) {
		throw NotImplementedException("LLM adapter cannot push down correlated or generated column reference");
	}
	auto column_index = column_ids[binding_index].GetPrimaryIndex();
	if (column_index >= get.names.size()) {
		throw NotImplementedException("LLM adapter cannot push down virtual column reference");
	}
	auto result = doc.Object();
	JsonAdd(doc, result, "kind", doc.String("column"));
	JsonAdd(doc, result, "name", doc.String(get.names[column_index]));
	JsonAdd(doc, result, "duckdb_type", doc.String(get.returned_types[column_index].ToString()));
	return result;
}

static yyjson_mut_val *LiteralToJson(JsonWriteDocument &doc, const BoundConstantExpression &constant) {
	auto result = doc.Object();
	JsonAdd(doc, result, "kind", doc.String("literal"));
	JsonAdd(doc, result, "value", ValueToJson(doc, constant.value));
	JsonAdd(doc, result, "duckdb_type", doc.String(constant.value.type().ToString()));
	return result;
}

static string ComparisonOperatorToString(ExpressionType type) {
	switch (type) {
	case ExpressionType::COMPARE_EQUAL:
	case ExpressionType::COMPARE_NOTEQUAL:
	case ExpressionType::COMPARE_LESSTHAN:
	case ExpressionType::COMPARE_GREATERTHAN:
	case ExpressionType::COMPARE_LESSTHANOREQUALTO:
	case ExpressionType::COMPARE_GREATERTHANOREQUALTO:
		return ExpressionTypeToOperator(type);
	default:
		throw NotImplementedException("LLM adapter cannot push down comparison \"%s\" yet", ExpressionTypeToString(type));
	}
}

static bool IsSupportedArithmeticFunction(const string &name) {
	return name == "+" || name == "-" || name == "*" || name == "/" || name == "//" || name == "%";
}

static yyjson_mut_val *ExpressionToPredicate(JsonWriteDocument &doc, const LogicalGet &get, const Expression &expr);

static yyjson_mut_val *ArithmeticFunctionToJson(JsonWriteDocument &doc, const LogicalGet &get,
                                                const BoundFunctionExpression &function) {
	if (!IsSupportedArithmeticFunction(function.function.name)) {
		throw NotImplementedException("LLM adapter cannot push down scalar function \"%s\" yet", function.function.name);
	}
	if (function.children.empty() || function.children.size() > 2) {
		throw NotImplementedException("LLM adapter cannot push down arithmetic function \"%s\" with %llu arguments",
		                              function.function.name, function.children.size());
	}
	if (!IsSupportedAdapterType(function.return_type)) {
		throw NotImplementedException("LLM adapter cannot push down arithmetic result type \"%s\" yet",
		                              function.return_type.ToString());
	}
	auto args = doc.Array();
	for (auto &child : function.children) {
		JsonAppend(args, ExpressionToPredicate(doc, get, *child));
	}
	auto result = doc.Object();
	JsonAdd(doc, result, "kind", doc.String("arithmetic"));
	JsonAdd(doc, result, "op", doc.String(function.function.name));
	JsonAdd(doc, result, "duckdb_type", doc.String(function.return_type.ToString()));
	JsonAdd(doc, result, "args", args);
	return result;
}

static yyjson_mut_val *ExpressionToPredicate(JsonWriteDocument &doc, const LogicalGet &get, const Expression &expr) {
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_COLUMN_REF:
		return ColumnReferenceToJson(doc, get, expr.Cast<BoundColumnRefExpression>());
	case ExpressionClass::BOUND_CONSTANT:
		return LiteralToJson(doc, expr.Cast<BoundConstantExpression>());
	case ExpressionClass::BOUND_CAST: {
		auto &cast = expr.Cast<BoundCastExpression>();
		if (cast.try_cast) {
			throw NotImplementedException("LLM adapter cannot push down TRY_CAST predicates yet");
		}
		return ExpressionToPredicate(doc, get, *cast.child);
	}
	case ExpressionClass::BOUND_COMPARISON: {
		auto &comparison = expr.Cast<BoundComparisonExpression>();
		auto result = doc.Object();
		JsonAdd(doc, result, "kind", doc.String("comparison"));
		JsonAdd(doc, result, "op", doc.String(ComparisonOperatorToString(expr.GetExpressionType())));
		JsonAdd(doc, result, "left", ExpressionToPredicate(doc, get, *comparison.left));
		JsonAdd(doc, result, "right", ExpressionToPredicate(doc, get, *comparison.right));
		return result;
	}
	case ExpressionClass::BOUND_CONJUNCTION: {
		auto &conjunction = expr.Cast<BoundConjunctionExpression>();
		auto children = doc.Array();
		for (auto &child : conjunction.children) {
			JsonAppend(children, ExpressionToPredicate(doc, get, *child));
		}
		auto result = doc.Object();
		JsonAdd(doc, result, "kind", doc.String(expr.GetExpressionType() == ExpressionType::CONJUNCTION_AND ? "and" : "or"));
		JsonAdd(doc, result, "children", children);
		return result;
	}
	case ExpressionClass::BOUND_FUNCTION:
		return ArithmeticFunctionToJson(doc, get, expr.Cast<BoundFunctionExpression>());
	case ExpressionClass::BOUND_OPERATOR: {
		auto &op = expr.Cast<BoundOperatorExpression>();
		if (expr.GetExpressionType() == ExpressionType::OPERATOR_IS_NULL ||
		    expr.GetExpressionType() == ExpressionType::OPERATOR_IS_NOT_NULL) {
			if (op.children.size() != 1) {
				throw InternalException("Unexpected IS NULL child count");
			}
			auto result = doc.Object();
			JsonAdd(doc, result, "kind",
			        doc.String(expr.GetExpressionType() == ExpressionType::OPERATOR_IS_NULL ? "is_null" : "is_not_null"));
			JsonAdd(doc, result, "expr", ExpressionToPredicate(doc, get, *op.children[0]));
			return result;
		}
		throw NotImplementedException("LLM adapter cannot push down operator \"%s\" yet",
		                              ExpressionTypeToString(expr.GetExpressionType()));
	}
	default:
		throw NotImplementedException("LLM adapter cannot push down predicate expression \"%s\" yet", expr.ToString());
	}
}

class LlmAdapterClient {
public:
	LlmAdapterClient(string endpoint_p, string checkpoint_ref_p)
	    : endpoint(NormalizeEndpoint(std::move(endpoint_p))), parsed_endpoint(ParseHttpEndpoint(endpoint)),
	      checkpoint_ref(std::move(checkpoint_ref_p)) {
	}
	const string &CheckpointRef() const {
		return checkpoint_ref;
	}
	unique_ptr<JsonReadDocument> Post(ClientContext &, const string &path, const string &body) const {
		return make_uniq<JsonReadDocument>(HttpPostJson(parsed_endpoint, path, body), path);
	}

private:
	string endpoint;
	ParsedHttpEndpoint parsed_endpoint;
	string checkpoint_ref;
};

struct LlmScanBindData : public FunctionData {
	LlmScanBindData(LlmCatalog &catalog_p, LlmTableEntry &table_p) : catalog(catalog_p), table(table_p) {
	}
	LlmCatalog &catalog;
	LlmTableEntry &table;
	string predicate_json;
	bool has_predicate = false;
	bool has_limit = false;
	idx_t limit = 0;
	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<LlmScanBindData>(catalog, table);
		result->predicate_json = predicate_json;
		result->has_predicate = has_predicate;
		result->has_limit = has_limit;
		result->limit = limit;
		return std::move(result);
	}
	bool Equals(const FunctionData &other) const override {
		return this == &other;
	}
	bool SupportStatementCache() const override {
		return false;
	}
};

struct LlmScanGlobalState : public GlobalTableFunctionState {
	vector<vector<Value>> rows;
	vector<LogicalType> response_types;
	vector<LogicalType> output_types;
	vector<idx_t> output_to_response;
	idx_t offset = 0;
	idx_t MaxThreads() const override {
		return 1;
	}
};

class LlmClientState : public ClientContextState {
public:
	void MarkAttached() {
		attached = true;
	}
	void TransactionBegin(MetaTransaction &, ClientContext &context) override {
		if (attached && !context.transaction.IsAutoCommit()) {
			throw TransactionException("LLM catalogs do not support explicit transactions; use autocommit statements");
		}
	}
	bool CanRequestRebind() override {
		return true;
	}
	RebindQueryInfo OnFinalizePrepare(ClientContext &, PreparedStatementData &prepared_statement,
	                                  PreparedStatementMode) override {
		if (attached && prepared_statement.statement_type == StatementType::TRANSACTION_STATEMENT) {
			throw TransactionException("LLM catalogs do not support explicit transactions; use autocommit statements");
		}
		return RebindQueryInfo::DO_NOT_REBIND;
	}

private:
	bool attached = false;
};

static void MarkLlmAttached(ClientContext &context) {
	auto state = context.registered_state->GetOrCreate<LlmClientState>("llm_extension_state");
	state->MarkAttached();
}

class LlmTransaction : public Transaction {
public:
	LlmTransaction(TransactionManager &manager, ClientContext &context) : Transaction(manager, context) {
	}
};

class LlmTransactionManager : public TransactionManager {
public:
	explicit LlmTransactionManager(AttachedDatabase &db_p) : TransactionManager(db_p) {
	}
	Transaction &StartTransaction(ClientContext &context) override {
		auto transaction = make_uniq<LlmTransaction>(*this, context);
		auto &result = *transaction;
		lock_guard<mutex> guard(transaction_lock);
		transactions[result] = std::move(transaction);
		return result;
	}
	ErrorData CommitTransaction(ClientContext &, Transaction &transaction) override {
		lock_guard<mutex> guard(transaction_lock);
		transactions.erase(transaction);
		return ErrorData();
	}
	void RollbackTransaction(Transaction &transaction) override {
		lock_guard<mutex> guard(transaction_lock);
		transactions.erase(transaction);
	}
	void Checkpoint(ClientContext &, bool force = false) override {
	}

private:
	mutex transaction_lock;
	reference_map_t<Transaction, unique_ptr<LlmTransaction>> transactions;
};

class LlmTableEntry : public TableCatalogEntry {
public:
	LlmTableEntry(LlmCatalog &llm_catalog_p, SchemaCatalogEntry &schema, CreateTableInfo &info, LlmTableMeta meta_p);
	unique_ptr<BaseStatistics> GetStatistics(ClientContext &, column_t) override {
		return nullptr;
	}
	TableStorageInfo GetStorageInfo(ClientContext &) override {
		return TableStorageInfo();
	}
	TableFunction GetScanFunction(ClientContext &, unique_ptr<FunctionData> &bind_data) override;
	const LlmTableMeta &GetMeta() const {
		return meta;
	}

private:
	LlmCatalog &llm_catalog;
	LlmTableMeta meta;
};

class LlmSchemaEntry : public SchemaCatalogEntry {
public:
	LlmSchemaEntry(LlmCatalog &catalog, CreateSchemaInfo &info);
	optional_ptr<CatalogEntry> CreateTable(CatalogTransaction transaction, BoundCreateTableInfo &info) override;
	optional_ptr<CatalogEntry> CreateFunction(CatalogTransaction transaction, CreateFunctionInfo &info) override {
		throw BinderException("LLM catalog does not support creating functions");
	}
	optional_ptr<CatalogEntry> CreateIndex(CatalogTransaction transaction, CreateIndexInfo &info,
	                                       TableCatalogEntry &table) override {
		throw BinderException("LLM catalog does not support creating indexes");
	}
	optional_ptr<CatalogEntry> CreateView(CatalogTransaction transaction, CreateViewInfo &info) override {
		throw BinderException("LLM catalog does not support creating views");
	}
	optional_ptr<CatalogEntry> CreateSequence(CatalogTransaction transaction, CreateSequenceInfo &info) override {
		throw BinderException("LLM catalog does not support creating sequences");
	}
	optional_ptr<CatalogEntry> CreateTableFunction(CatalogTransaction transaction,
	                                               CreateTableFunctionInfo &info) override {
		throw BinderException("LLM catalog does not support creating table functions");
	}
	optional_ptr<CatalogEntry> CreateCopyFunction(CatalogTransaction transaction, CreateCopyFunctionInfo &info) override {
		throw BinderException("LLM catalog does not support creating copy functions");
	}
	optional_ptr<CatalogEntry> CreatePragmaFunction(CatalogTransaction transaction,
	                                                CreatePragmaFunctionInfo &info) override {
		throw BinderException("LLM catalog does not support creating pragma functions");
	}
	optional_ptr<CatalogEntry> CreateCollation(CatalogTransaction transaction, CreateCollationInfo &info) override {
		throw BinderException("LLM catalog does not support creating collations");
	}
	optional_ptr<CatalogEntry> CreateType(CatalogTransaction transaction, CreateTypeInfo &info) override {
		throw BinderException("LLM catalog does not support creating types");
	}
	void Alter(CatalogTransaction transaction, AlterInfo &info) override {
		throw BinderException("LLM catalog does not support ALTER yet");
	}
	void Scan(ClientContext &context, CatalogType type, const std::function<void(CatalogEntry &)> &callback) override {
		Scan(type, callback);
	}
	void Scan(CatalogType type, const std::function<void(CatalogEntry &)> &callback) override {
		if (type != CatalogType::TABLE_ENTRY && type != CatalogType::INVALID) {
			return;
		}
		for (auto &entry : tables) {
			callback(*entry.second);
		}
	}
	void DropEntry(ClientContext &context, DropInfo &info) override {
		throw NotImplementedException("LLM DROP is not implemented in the adapter protocol yet");
	}
	optional_ptr<CatalogEntry> LookupEntry(CatalogTransaction transaction, const EntryLookupInfo &lookup_info) override {
		auto entry = tables.find(lookup_info.GetEntryName());
		return entry == tables.end() ? nullptr : entry->second.get();
	}
	void ReplaceTables(ClientContext &context, const vector<LlmTableMeta> &new_tables);

private:
	unique_ptr<CreateTableInfo> BuildCreateInfo(const LlmTableMeta &table);
	LlmCatalog &llm_catalog;
	case_insensitive_map_t<unique_ptr<LlmTableEntry>> tables;
};

class LlmCatalog : public Catalog {
public:
	LlmCatalog(AttachedDatabase &db_p, string path_p, string endpoint_p)
	    : Catalog(db_p), path(std::move(path_p)), client(std::move(endpoint_p), path), catalog_version("v0") {
	}
	void Initialize(bool load_builtin) override {
		CreateSchemaInfo info;
		info.catalog = GetName();
		info.schema = DEFAULT_SCHEMA;
		main_schema = make_uniq<LlmSchemaEntry>(*this, info);
	}
	void Initialize(optional_ptr<ClientContext> context, bool load_builtin) override {
		Initialize(load_builtin);
		if (context) {
			auto &context_ref = *context.get();
			ReplaceCatalog(context_ref, Introspect(context_ref));
		}
	}
	string GetCatalogType() override {
		return "llm";
	}
	optional_ptr<CatalogEntry> CreateSchema(CatalogTransaction transaction, CreateSchemaInfo &info) override {
		throw BinderException("LLM catalog does not support creating schemas");
	}
	void ScanSchemas(ClientContext &context, std::function<void(SchemaCatalogEntry &)> callback) override {
		callback(*main_schema);
	}
	optional_ptr<SchemaCatalogEntry> LookupSchema(CatalogTransaction transaction, const EntryLookupInfo &schema_lookup,
	                                              OnEntryNotFound if_not_found) override {
		auto &schema_name = schema_lookup.GetEntryName();
		if (schema_name == DEFAULT_SCHEMA || schema_name == INVALID_SCHEMA) {
			return main_schema.get();
		}
		if (if_not_found == OnEntryNotFound::RETURN_NULL) {
			return nullptr;
		}
		throw BinderException("LLM catalogs only expose the \"%s\" schema", string(DEFAULT_SCHEMA));
	}
	PhysicalOperator &PlanCreateTableAs(ClientContext &context, PhysicalPlanGenerator &planner, LogicalCreateTable &op,
	                                    PhysicalOperator &plan) override {
		throw NotImplementedException("LLM CREATE TABLE AS is not implemented in the adapter protocol yet");
	}
	PhysicalOperator &PlanInsert(ClientContext &context, PhysicalPlanGenerator &planner, LogicalInsert &op,
	                             optional_ptr<PhysicalOperator> plan) override {
		throw NotImplementedException("LLM INSERT is not implemented in the adapter protocol yet");
	}
	PhysicalOperator &PlanDelete(ClientContext &context, PhysicalPlanGenerator &planner, LogicalDelete &op,
	                             PhysicalOperator &plan) override {
		throw NotImplementedException("LLM DELETE is not implemented in the adapter protocol yet");
	}
	PhysicalOperator &PlanUpdate(ClientContext &context, PhysicalPlanGenerator &planner, LogicalUpdate &op,
	                             PhysicalOperator &plan) override {
		throw NotImplementedException("LLM UPDATE is not implemented in the adapter protocol yet");
	}
	DatabaseSize GetDatabaseSize(ClientContext &context) override {
		return DatabaseSize();
	}
	bool InMemory() override {
		return true;
	}
	string GetDBPath() override {
		return path;
	}
	const string &CatalogVersion() const {
		return catalog_version;
	}
	LlmCatalogSnapshot Introspect(ClientContext &context) {
		JsonWriteDocument doc;
		auto request = doc.Object();
		JsonAdd(doc, request, "type", doc.String("introspect_catalog"));
		JsonAdd(doc, request, "catalog", doc.String(GetName()));
		JsonAdd(doc, request, "checkpoint_ref", doc.String(client.CheckpointRef()));
		auto response = client.Post(context, "/v1/catalog/introspect", doc.Write(request));
		return ParseCatalogSnapshot(context, response->Root());
	}
	void ReplaceCatalog(ClientContext &context, const LlmCatalogSnapshot &snapshot) {
		catalog_version = snapshot.version;
		main_schema->ReplaceTables(context, snapshot.tables);
	}
	optional_ptr<CatalogEntry> ApplyCreateTable(CatalogTransaction transaction, BoundCreateTableInfo &info);
	unique_ptr<JsonReadDocument> Select(ClientContext &context, const string &query_json) {
		JsonWriteDocument doc;
		auto request = doc.Object();
		JsonAdd(doc, request, "type", doc.String("select"));
		JsonAdd(doc, request, "catalog_version", doc.String(catalog_version));
		JsonAdd(doc, request, "query", doc.ParseAndCopy(query_json, "select query"));
		return client.Post(context, "/v1/query/select", doc.Write(request));
	}

private:
	void DropSchema(ClientContext &context, DropInfo &info) override {
		throw BinderException("LLM catalog does not support dropping schemas");
	}
	LlmCatalogSnapshot ParseCatalogSnapshot(ClientContext &context, yyjson_val *json) {
		LlmCatalogSnapshot snapshot;
		snapshot.version = JsonString(JsonRequire(json, "catalog_version", "catalog snapshot"), "catalog_version");
		auto schemas = JsonArray(JsonRequire(json, "schemas", "catalog snapshot"), "catalog snapshot schemas");
		size_t schema_idx, schema_max;
		yyjson_val *schema_json;
		yyjson_arr_foreach(schemas, schema_idx, schema_max, schema_json) {
			auto schema_name = JsonString(JsonRequire(schema_json, "name", "schema"), "schema.name");
			if (schema_name != DEFAULT_SCHEMA) {
				throw NotImplementedException("LLM adapter only supports the \"%s\" schema for now",
				                              string(DEFAULT_SCHEMA));
			}
			auto tables = JsonArray(JsonRequire(schema_json, "tables", "schema"), "schema.tables");
			size_t table_idx, table_max;
			yyjson_val *table_json;
			yyjson_arr_foreach(tables, table_idx, table_max, table_json) {
				LlmTableMeta table;
				table.schema = schema_name;
				table.name = JsonString(JsonRequire(table_json, "name", "table"), "table.name");
				auto columns = JsonArray(JsonRequire(table_json, "columns", "table"), "table.columns");
				size_t column_idx, column_max;
				yyjson_val *column_json;
				yyjson_arr_foreach(columns, column_idx, column_max, column_json) {
					LlmColumnMeta column;
					column.name = JsonString(JsonRequire(column_json, "name", "column"), "column.name");
					auto type_string = JsonString(JsonRequire(column_json, "duckdb_type", "column"), "column.duckdb_type");
					column.type = ParseAdapterType(context, type_string);
					column.nullable = JsonBoolean(JsonRequire(column_json, "nullable", "column"), "column.nullable");
					table.columns.push_back(std::move(column));
				}
				if (auto primary_key = JsonGet(table_json, "primary_key")) {
					auto primary_key_array = JsonArray(primary_key, "table.primary_key");
					size_t key_idx, key_max;
					yyjson_val *key_json;
					yyjson_arr_foreach(primary_key_array, key_idx, key_max, key_json) {
						table.primary_key.push_back(JsonString(key_json, "primary_key column"));
					}
				}
				snapshot.tables.push_back(std::move(table));
			}
		}
		return snapshot;
	}
	string path;
	LlmAdapterClient client;
	string catalog_version;
	unique_ptr<LlmSchemaEntry> main_schema;
};

LlmTableEntry::LlmTableEntry(LlmCatalog &llm_catalog_p, SchemaCatalogEntry &schema, CreateTableInfo &info,
                             LlmTableMeta meta_p)
    : TableCatalogEntry(static_cast<Catalog &>(llm_catalog_p), schema, info), llm_catalog(llm_catalog_p),
      meta(std::move(meta_p)) {
}

LlmSchemaEntry::LlmSchemaEntry(LlmCatalog &catalog, CreateSchemaInfo &info)
    : SchemaCatalogEntry(static_cast<Catalog &>(catalog), info), llm_catalog(catalog) {
}

unique_ptr<CreateTableInfo> LlmSchemaEntry::BuildCreateInfo(const LlmTableMeta &table) {
	auto info = make_uniq<CreateTableInfo>();
	info->catalog = llm_catalog.GetName();
	info->schema = table.schema;
	info->table = table.name;
	for (auto &column : table.columns) {
		info->columns.AddColumn(ColumnDefinition(column.name, column.type));
	}
	for (idx_t column_idx = 0; column_idx < table.columns.size(); column_idx++) {
		if (!table.columns[column_idx].nullable) {
			info->constraints.push_back(make_uniq<NotNullConstraint>(LogicalIndex(column_idx)));
		}
	}
	if (!table.primary_key.empty()) {
		info->constraints.push_back(make_uniq<UniqueConstraint>(table.primary_key, true));
	}
	return info;
}

void LlmSchemaEntry::ReplaceTables(ClientContext &context, const vector<LlmTableMeta> &new_tables) {
	tables.clear();
	for (auto &table : new_tables) {
		auto info = BuildCreateInfo(table);
		auto entry = make_uniq<LlmTableEntry>(llm_catalog, *this, *info, table);
		tables[table.name] = std::move(entry);
	}
}

static bool ConstraintColumnMatches(const UniqueConstraint &constraint, const ColumnList &columns, idx_t column_idx) {
	if (constraint.HasIndex()) {
		return constraint.GetIndex().index == column_idx;
	}
	for (auto &name : constraint.GetColumnNames()) {
		if (columns.GetColumn(name).Logical().index == column_idx) {
			return true;
		}
	}
	return false;
}

static bool ColumnIsNullable(const CreateTableInfo &base, idx_t column_idx) {
	for (auto &constraint : base.constraints) {
		if (constraint->type == ConstraintType::NOT_NULL &&
		    constraint->Cast<NotNullConstraint>().index.index == column_idx) {
			return false;
		}
		if (constraint->type == ConstraintType::UNIQUE) {
			auto &unique = constraint->Cast<UniqueConstraint>();
			if (unique.IsPrimaryKey() && ConstraintColumnMatches(unique, base.columns, column_idx)) {
				return false;
			}
		}
	}
	return true;
}

static vector<string> ExtractPrimaryKey(const CreateTableInfo &base) {
	vector<string> result;
	for (auto &constraint : base.constraints) {
		if (constraint->type != ConstraintType::UNIQUE) {
			continue;
		}
		auto &unique = constraint->Cast<UniqueConstraint>();
		if (!unique.IsPrimaryKey()) {
			throw NotImplementedException("LLM adapter does not support UNIQUE constraints yet");
		}
		if (unique.HasIndex()) {
			result.push_back(base.columns.GetColumn(unique.GetIndex()).Name());
		} else {
			for (auto &name : unique.GetColumnNames()) {
				result.push_back(name);
			}
		}
	}
	return result;
}

static void ValidateCreateTable(const CreateTableInfo &base) {
	if (base.on_conflict != OnCreateConflict::ERROR_ON_CONFLICT) {
		throw NotImplementedException("LLM adapter only supports CREATE TABLE with error-on-conflict semantics");
	}
	if (base.query) {
		throw NotImplementedException("LLM CREATE TABLE AS is not implemented in the adapter protocol yet");
	}
	if (!base.partition_keys.empty() || !base.sort_keys.empty() || !base.options.empty()) {
		throw NotImplementedException("LLM adapter does not support table options, partitioning, or sorting yet");
	}
	for (auto &column : base.columns.Logical()) {
		if (!IsSupportedAdapterType(column.Type())) {
			throw NotImplementedException("LLM adapter does not support DuckDB type \"%s\" yet", column.Type().ToString());
		}
		if (column.HasDefaultValue()) {
			throw NotImplementedException("LLM adapter does not support column defaults yet");
		}
		if (column.Generated()) {
			throw NotImplementedException("LLM adapter does not support generated columns yet");
		}
	}
	for (auto &constraint : base.constraints) {
		if (constraint->type == ConstraintType::NOT_NULL) {
			continue;
		}
		if (constraint->type == ConstraintType::UNIQUE && constraint->Cast<UniqueConstraint>().IsPrimaryKey()) {
			continue;
		}
		throw NotImplementedException("LLM adapter does not support constraint \"%s\" yet", constraint->ToString());
	}
}

static yyjson_mut_val *CreateTableOpToJson(JsonWriteDocument &doc, const string &catalog_name,
                                           BoundCreateTableInfo &info) {
	auto &base = info.Base();
	ValidateCreateTable(base);
	auto columns = doc.Array();
	idx_t column_idx = 0;
	for (auto &column : base.columns.Logical()) {
		auto column_json = doc.Object();
		JsonAdd(doc, column_json, "name", doc.String(column.Name()));
		JsonAdd(doc, column_json, "duckdb_type", doc.String(column.Type().ToString()));
		JsonAdd(doc, column_json, "nullable", doc.Bool(ColumnIsNullable(base, column_idx)));
		JsonAdd(doc, column_json, "default", doc.Null());
		JsonAdd(doc, column_json, "generated", doc.Bool(false));
		JsonAppend(columns, column_json);
		column_idx++;
	}
	auto primary_key = doc.Array();
	for (auto &column : ExtractPrimaryKey(base)) {
		JsonAppend(primary_key, doc.String(column));
	}
	auto result = doc.Object();
	JsonAdd(doc, result, "op", doc.String("create_table"));
	JsonAdd(doc, result, "catalog", doc.String(catalog_name));
	JsonAdd(doc, result, "schema", doc.String(base.schema));
	JsonAdd(doc, result, "table", doc.String(base.table));
	JsonAdd(doc, result, "on_conflict", doc.String("error"));
	JsonAdd(doc, result, "columns", columns);
	JsonAdd(doc, result, "primary_key", primary_key);
	JsonAdd(doc, result, "unique", doc.Array());
	JsonAdd(doc, result, "checks", doc.Array());
	JsonAdd(doc, result, "foreign_keys", doc.Array());
	return result;
}

optional_ptr<CatalogEntry> LlmCatalog::ApplyCreateTable(CatalogTransaction transaction, BoundCreateTableInfo &info) {
	if (!transaction.HasContext()) {
		throw InternalException("LLM CREATE TABLE requires a client context");
	}
	auto &context = transaction.GetContext();
	JsonWriteDocument doc;
	auto op = CreateTableOpToJson(doc, GetName(), info);
	auto operations = doc.Array();
	JsonAppend(operations, op);
	auto request = doc.Object();
	JsonAdd(doc, request, "type", doc.String("apply_mutation"));
	JsonAdd(doc, request, "base_catalog_version", doc.String(catalog_version));
	JsonAdd(doc, request, "operations", operations);
	auto response = client.Post(context, "/v1/mutations/apply", doc.Write(request));
	auto root = response->Root();
	auto status = JsonString(JsonRequire(root, "status", "mutation response"), "mutation status");
	if (status != "applied") {
		throw IOException("LLM mutation failed with status \"%s\"", status);
	}
	auto snapshot = ParseCatalogSnapshot(context, JsonRequire(root, "catalog", "mutation response"));
	ReplaceCatalog(context, snapshot);
	return main_schema->LookupEntry(transaction, EntryLookupInfo(CatalogType::TABLE_ENTRY, info.Base().table));
}

optional_ptr<CatalogEntry> LlmSchemaEntry::CreateTable(CatalogTransaction transaction, BoundCreateTableInfo &info) {
	return llm_catalog.ApplyCreateTable(transaction, info);
}

static vector<idx_t> BuildOutputColumnIds(const vector<ColumnIndex> &column_ids, const vector<idx_t> &projection_ids) {
	vector<idx_t> result;
	auto append_column = [&](idx_t column_id_index) {
		auto column_id = column_ids[column_id_index].GetPrimaryIndex();
		result.push_back(column_id);
	};
	if (projection_ids.empty()) {
		for (idx_t i = 0; i < column_ids.size(); i++) {
			append_column(i);
		}
	} else {
		for (auto projection_id : projection_ids) {
			append_column(projection_id);
		}
	}
	return result;
}

static yyjson_mut_val *BuildProjectionJson(JsonWriteDocument &doc, const LlmTableEntry &table,
                                           const vector<idx_t> &output_column_ids,
                                           vector<LogicalType> &response_types, vector<LogicalType> &output_types,
                                           vector<idx_t> &output_to_response) {
	auto &meta = table.GetMeta();
	vector<idx_t> response_column_ids;
	for (auto column_id : output_column_ids) {
		if (column_id >= meta.columns.size()) {
			throw NotImplementedException("LLM adapter does not support virtual or row-id columns");
		}
		output_types.push_back(meta.columns[column_id].type);
		if (std::find(response_column_ids.begin(), response_column_ids.end(), column_id) == response_column_ids.end()) {
			response_column_ids.push_back(column_id);
		}
	}
	std::sort(response_column_ids.begin(), response_column_ids.end());

	auto projections = doc.Array();
	for (auto column_id : response_column_ids) {
		auto &column = meta.columns[column_id];
		response_types.push_back(column.type);
		auto projection = doc.Object();
		JsonAdd(doc, projection, "name", doc.String(column.name));
		JsonAdd(doc, projection, "duckdb_type", doc.String(column.type.ToString()));
		JsonAppend(projections, projection);
	}
	for (auto column_id : output_column_ids) {
		auto entry = std::find(response_column_ids.begin(), response_column_ids.end(), column_id);
		if (entry == response_column_ids.end()) {
			throw InternalException("LLM scan output column was not requested from adapter");
		}
		output_to_response.push_back(NumericCast<idx_t>(entry - response_column_ids.begin()));
	}
	return projections;
}

static unique_ptr<GlobalTableFunctionState> LlmScanInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind = input.bind_data->Cast<LlmScanBindData>();
	if (input.filters && !input.filters->filters.empty()) {
		throw NotImplementedException(
		    "LLM adapter cannot apply filters locally; the predicate was not pushed down to Python");
	}
	auto result = make_uniq<LlmScanGlobalState>();
	auto output_column_ids = BuildOutputColumnIds(input.column_indexes, input.projection_ids);
	JsonWriteDocument doc;
	auto projection = BuildProjectionJson(doc, bind.table, output_column_ids, result->response_types, result->output_types,
	                                      result->output_to_response);
	auto query = doc.Object();
	JsonAdd(doc, query, "schema", doc.String(bind.table.GetMeta().schema));
	JsonAdd(doc, query, "table", doc.String(bind.table.GetMeta().name));
	JsonAdd(doc, query, "projection", projection);
	JsonAdd(doc, query, "predicate", bind.has_predicate ? doc.ParseAndCopy(bind.predicate_json, "select predicate")
	                                                    : doc.Null());
	JsonAdd(doc, query, "limit", bind.has_limit ? doc.Int(NumericCast<int64_t>(bind.limit)) : doc.Null());
	auto response = bind.catalog.Select(context, doc.Write(query));
	auto root = response->Root();
	auto columns = JsonArray(JsonRequire(root, "columns", "select response"), "select columns");
	if (yyjson_arr_size(columns) != result->response_types.size()) {
		throw IOException("LLM adapter select returned %llu columns, expected %llu",
		                  NumericCast<idx_t>(yyjson_arr_size(columns)),
		                  result->response_types.size());
	}
	size_t column_idx, column_max;
	yyjson_val *column_json;
	yyjson_arr_foreach(columns, column_idx, column_max, column_json) {
		auto returned_type = JsonString(JsonRequire(column_json, "duckdb_type", "select column"), "select column type");
		if (returned_type != result->response_types[column_idx].ToString()) {
			throw IOException("LLM adapter select returned type \"%s\" for column %llu, expected \"%s\"", returned_type,
			                  column_idx, result->response_types[column_idx].ToString());
		}
	}
	auto rows = JsonArray(JsonRequire(root, "rows", "select response"), "select rows");
	size_t row_idx, row_max;
	yyjson_val *row_json;
	yyjson_arr_foreach(rows, row_idx, row_max, row_json) {
		auto row_array = JsonArray(row_json, "select row");
		if (yyjson_arr_size(row_array) != result->response_types.size()) {
			throw IOException("LLM adapter select returned a row with the wrong width");
		}
		vector<Value> row_values;
		size_t value_idx, value_max;
		yyjson_val *value_json;
		yyjson_arr_foreach(row_array, value_idx, value_max, value_json) {
			row_values.push_back(JsonToValue(value_json, result->response_types[value_idx], "select row value"));
		}
		result->rows.push_back(std::move(row_values));
	}
	return std::move(result);
}

static void LlmScanFunction(ClientContext &, TableFunctionInput &data, DataChunk &output) {
	auto &state = data.global_state->Cast<LlmScanGlobalState>();
	if (state.offset >= state.rows.size()) {
		return;
	}
	auto count = MinValue<idx_t>(STANDARD_VECTOR_SIZE, state.rows.size() - state.offset);
	output.SetCardinality(count);
	for (idx_t row_idx = 0; row_idx < count; row_idx++) {
		auto &row = state.rows[state.offset + row_idx];
		for (idx_t col_idx = 0; col_idx < state.output_types.size(); col_idx++) {
			auto response_col_idx = state.output_to_response[col_idx];
			output.SetValue(col_idx, row_idx, row[response_col_idx]);
		}
	}
	state.offset += count;
}

static void LlmPushdownComplexFilter(ClientContext &, LogicalGet &get, FunctionData *bind_data,
                                     vector<unique_ptr<Expression>> &filters) {
	if (filters.empty()) {
		return;
	}
	auto &bind = bind_data->Cast<LlmScanBindData>();
	JsonWriteDocument doc;
	auto predicates = doc.Array();
	for (auto &filter : filters) {
		JsonAppend(predicates, ExpressionToPredicate(doc, get, *filter));
	}
	bind.has_predicate = true;
	yyjson_mut_val *predicate = nullptr;
	if (yyjson_mut_arr_size(predicates) == 1) {
		predicate = yyjson_mut_arr_get(predicates, 0);
	} else {
		predicate = doc.Object();
		JsonAdd(doc, predicate, "kind", doc.String("and"));
		JsonAdd(doc, predicate, "children", predicates);
	}
	bind.predicate_json = doc.Write(predicate);
	filters.clear();
}

TableFunction LlmTableEntry::GetScanFunction(ClientContext &, unique_ptr<FunctionData> &bind_data) {
	bind_data = make_uniq<LlmScanBindData>(llm_catalog, *this);
	TableFunction scan("llm_scan", {}, LlmScanFunction, nullptr, LlmScanInitGlobal, nullptr);
	scan.projection_pushdown = true;
	scan.filter_prune = true;
	scan.pushdown_complex_filter = LlmPushdownComplexFilter;
	scan.verify_serialization = false;
	return scan;
}

class LlmOptimizerExtension : public OptimizerExtension {
public:
	LlmOptimizerExtension() {
		optimize_function = Optimize;
	}

private:
	static bool IsLlmGet(LogicalOperator &op) {
		return op.type == LogicalOperatorType::LOGICAL_GET && op.Cast<LogicalGet>().function.name == "llm_scan";
	}
	static idx_t CountLlmGets(LogicalOperator &op) {
		idx_t count = IsLlmGet(op) ? 1 : 0;
		for (auto &child : op.children) {
			count += CountLlmGets(*child);
		}
		return count;
	}
	static void SetLimitOnLlmGets(LogicalOperator &op, idx_t limit) {
		if (IsLlmGet(op)) {
			auto &bind = op.Cast<LogicalGet>().bind_data->Cast<LlmScanBindData>();
			bind.has_limit = true;
			bind.limit = limit;
		}
		for (auto &child : op.children) {
			SetLimitOnLlmGets(*child, limit);
		}
	}
	static bool TryPushLimit(unique_ptr<LogicalOperator> &node) {
		if (node->type == LogicalOperatorType::LOGICAL_LIMIT) {
			auto &limit = node->Cast<LogicalLimit>();
			auto llm_gets = CountLlmGets(*node->children[0]);
			if (llm_gets == 0) {
				return false;
			}
			if (limit.limit_val.Type() != LimitNodeType::CONSTANT_VALUE) {
				throw NotImplementedException("LLM adapter only supports constant LIMIT pushdown");
			}
			if (limit.offset_val.Type() != LimitNodeType::UNSET &&
			    !(limit.offset_val.Type() == LimitNodeType::CONSTANT_VALUE &&
			      limit.offset_val.GetConstantValue() == 0)) {
				throw NotImplementedException("LLM adapter does not support OFFSET pushdown yet");
			}
			if (llm_gets > 1) {
				throw NotImplementedException("LLM adapter cannot push one LIMIT into multiple LLM scans");
			}
			SetLimitOnLlmGets(*node->children[0], limit.limit_val.GetConstantValue());
			node = std::move(node->children[0]);
			return true;
		}
		bool changed = false;
		for (auto &child : node->children) {
			changed = TryPushLimit(child) || changed;
		}
		return changed;
	}
	static void RejectLocalFilters(LogicalOperator &op) {
		if (op.type == LogicalOperatorType::LOGICAL_FILTER && !op.children.empty() && CountLlmGets(*op.children[0]) > 0) {
			throw NotImplementedException(
			    "LLM adapter cannot apply filters locally; the predicate was not pushed down to Python");
		}
		for (auto &child : op.children) {
			RejectLocalFilters(*child);
		}
	}
	static void Optimize(OptimizerExtensionInput &, unique_ptr<LogicalOperator> &plan) {
		TryPushLimit(plan);
		RejectLocalFilters(*plan);
	}
};

class LlmStorageExtension : public StorageExtension {
public:
	LlmStorageExtension() {
		attach = [](optional_ptr<StorageExtensionInfo>, ClientContext &context, AttachedDatabase &db, const string &,
		            AttachInfo &info, AttachOptions &) -> unique_ptr<Catalog> {
			MarkLlmAttached(context);
			auto endpoint = GetAttachStringOption(info, "endpoint", "http://0.0.0.0:5366");
			return make_uniq<LlmCatalog>(db, info.path, std::move(endpoint));
		};
		create_transaction_manager = [](optional_ptr<StorageExtensionInfo>, AttachedDatabase &db,
		                                Catalog &) -> unique_ptr<TransactionManager> {
			return make_uniq<LlmTransactionManager>(db);
		};
	}
};

static void LoadInternal(ExtensionLoader &loader) {
	auto &db = loader.GetDatabaseInstance();
	auto &config = DBConfig::GetConfig(db);
	StorageExtension::Register(config, "llm", make_shared_ptr<LlmStorageExtension>());
	OptimizerExtension::Register(config, LlmOptimizerExtension());
}

void LlmExtension::Load(ExtensionLoader &loader) {
	LoadInternal(loader);
}

std::string LlmExtension::Name() {
	return "llm";
}

std::string LlmExtension::Version() const {
#ifdef EXT_VERSION_LLM
	return EXT_VERSION_LLM;
#else
	return "";
#endif
}

} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(llm, loader) {
	duckdb::LoadInternal(loader);
}
}
