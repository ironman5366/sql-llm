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
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_create_table.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_update.hpp"
#include "duckdb/planner/parsed_data/bound_create_table_info.hpp"
#include "duckdb/planner/table_filter.hpp"
#include "duckdb/storage/database_size.hpp"
#include "duckdb/storage/storage_extension.hpp"
#include "duckdb/storage/table_storage_info.hpp"
#include "duckdb/common/progress_bar/display/terminal_progress_bar_display.hpp"
#include "duckdb/transaction/transaction.hpp"
#include "duckdb/transaction/transaction_manager.hpp"

#include "httplib.hpp"
#include "yyjson.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifndef _WIN32
#include <unistd.h>
#endif

namespace duckdb {
using namespace duckdb_yyjson; // NOLINT

class LlmCatalog;
class LlmSchemaEntry;
class LlmTableEntry;

static string WriteJsonAndFree(yyjson_mut_doc *doc) {
	size_t len = 0;
	yyjson_write_err error;
	auto data = yyjson_mut_write_opts(doc, YYJSON_WRITE_NOFLAG, nullptr, &len, &error);
	if (!data) {
		yyjson_mut_doc_free(doc);
		throw IOException("Failed to serialize LLM adapter JSON: %s", error.msg ? error.msg : "unknown error");
	}
	string result(data, len);
	std::free(data);
	yyjson_mut_doc_free(doc);
	return result;
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

static void HttpPostJsonLines(const ParsedHttpEndpoint &endpoint, const string &path, const string &body,
                              const std::function<bool(const string &)> &line_callback) {
	auto full_path = endpoint.path_prefix.empty() ? path : endpoint.path_prefix + path;
	duckdb_httplib::Client client(endpoint.proto_host_port);
	client.set_keep_alive(false);
	client.set_decompress(false);
	duckdb_httplib::Headers headers = {
	    {"Accept", "application/x-ndjson"},
	    {"X-SQL-LLM-Stream", "1"},
	};
	string pending;
	bool callback_ok = true;
	auto response = client.Post(
	    full_path, headers, body, "application/json",
	    [&](const char *data, size_t data_length) {
		    if (!callback_ok) {
			    return false;
		    }
		    pending.append(data, data_length);
		    while (true) {
			    auto newline = pending.find('\n');
			    if (newline == string::npos) {
				    break;
			    }
			    auto line = pending.substr(0, newline);
			    pending.erase(0, newline + 1);
			    if (!line.empty() && line.back() == '\r') {
				    line.pop_back();
			    }
			    if (!line.empty()) {
				    callback_ok = line_callback(line);
				    if (!callback_ok) {
					    return false;
				    }
			    }
		    }
		    return true;
	    });
	if (callback_ok && !pending.empty()) {
		callback_ok = line_callback(pending);
	}
	if (!response) {
		throw IOException("LLM adapter request to %s%s failed: %s", endpoint.proto_host_port, full_path,
		                  to_string(response.error()));
	}
	if (response->status < 200 || response->status >= 300) {
		throw IOException("LLM adapter request to %s%s failed with HTTP %d: %s", endpoint.proto_host_port, full_path,
		                  response->status, response->body);
	}
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

static yyjson_mut_val *DuckValueToYyjsonLiteral(yyjson_mut_doc *doc, const Value &value) {
	if (value.IsNull()) {
		return yyjson_mut_null(doc);
	}
	switch (value.type().id()) {
	case LogicalTypeId::BOOLEAN:
		return yyjson_mut_bool(doc, value.GetValue<bool>());
	case LogicalTypeId::TINYINT:
		return yyjson_mut_int(doc, static_cast<int64_t>(value.GetValue<int8_t>()));
	case LogicalTypeId::SMALLINT:
		return yyjson_mut_int(doc, static_cast<int64_t>(value.GetValue<int16_t>()));
	case LogicalTypeId::INTEGER:
		return yyjson_mut_int(doc, static_cast<int64_t>(value.GetValue<int32_t>()));
	case LogicalTypeId::BIGINT:
		return yyjson_mut_int(doc, value.GetValue<int64_t>());
	case LogicalTypeId::FLOAT:
		return yyjson_mut_real(doc, static_cast<double>(value.GetValue<float>()));
	case LogicalTypeId::DOUBLE:
		return yyjson_mut_real(doc, value.GetValue<double>());
	case LogicalTypeId::VARCHAR: {
		auto string_value = value.GetValue<string>();
		return yyjson_mut_strncpy(doc, string_value.c_str(), string_value.size());
	}
	default:
		throw NotImplementedException("LLM adapter cannot serialize literal of type \"%s\" yet", value.type().ToString());
	}
}

static Value YyjsonLiteralToDuckValue(yyjson_val *json, const LogicalType &type, const string &context) {
	if (yyjson_is_null(json)) {
		return Value(type);
	}
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN: {
		if (!yyjson_is_bool(json)) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected boolean for %s", context);
		}
		return Value::BOOLEAN(yyjson_get_bool(json));
	}
	case LogicalTypeId::TINYINT:
		if (!yyjson_is_int(json)) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected integer for %s", context);
		}
		return Value::TINYINT(NumericCast<int8_t>(yyjson_get_sint(json)));
	case LogicalTypeId::SMALLINT:
		if (!yyjson_is_int(json)) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected integer for %s", context);
		}
		return Value::SMALLINT(NumericCast<int16_t>(yyjson_get_sint(json)));
	case LogicalTypeId::INTEGER:
		if (!yyjson_is_int(json)) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected integer for %s", context);
		}
		return Value::INTEGER(NumericCast<int32_t>(yyjson_get_sint(json)));
	case LogicalTypeId::BIGINT:
		if (!yyjson_is_int(json)) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected integer for %s", context);
		}
		return Value::BIGINT(yyjson_get_sint(json));
	case LogicalTypeId::FLOAT: {
		if (!yyjson_is_int(json) && !yyjson_is_real(json)) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected number for %s", context);
		}
		auto number = yyjson_is_int(json) ? static_cast<double>(yyjson_get_sint(json)) : yyjson_get_real(json);
		return Value(static_cast<float>(number));
	}
	case LogicalTypeId::DOUBLE:
		if (!yyjson_is_int(json) && !yyjson_is_real(json)) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected number for %s", context);
		}
		return Value::DOUBLE(yyjson_is_int(json) ? static_cast<double>(yyjson_get_sint(json)) : yyjson_get_real(json));
	case LogicalTypeId::VARCHAR:
		if (!yyjson_is_str(json)) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected string for %s", context);
		}
		return Value(string(yyjson_get_str(json), yyjson_get_len(json)));
	default:
		throw NotImplementedException("LLM adapter cannot materialize values of type \"%s\" yet", type.ToString());
	}
}

static yyjson_mut_val *ColumnReferenceToJson(yyjson_mut_doc *doc, const LogicalGet &get,
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
	auto result = yyjson_mut_obj(doc);
	auto &name = get.names[column_index];
	auto type_string = get.returned_types[column_index].ToString();
	yyjson_mut_obj_add_str(doc, result, "kind", "column");
	yyjson_mut_obj_add_strncpy(doc, result, "name", name.c_str(), name.size());
	yyjson_mut_obj_add_strncpy(doc, result, "duckdb_type", type_string.c_str(), type_string.size());
	return result;
}

static yyjson_mut_val *BoundReferenceToJson(yyjson_mut_doc *doc, const LogicalGet &get,
                                            const BoundReferenceExpression &ref) {
	auto binding_index = NumericCast<idx_t>(ref.index);
	auto &column_ids = get.GetColumnIds();
	if (binding_index >= column_ids.size()) {
		throw NotImplementedException("LLM adapter cannot map update expression reference \"%s\"", ref.ToString());
	}
	auto column_index = column_ids[binding_index].GetPrimaryIndex();
	if (column_index >= get.names.size()) {
		throw NotImplementedException("LLM adapter cannot push down virtual column reference");
	}
	auto result = yyjson_mut_obj(doc);
	auto &name = get.names[column_index];
	auto type_string = get.returned_types[column_index].ToString();
	yyjson_mut_obj_add_str(doc, result, "kind", "column");
	yyjson_mut_obj_add_strncpy(doc, result, "name", name.c_str(), name.size());
	yyjson_mut_obj_add_strncpy(doc, result, "duckdb_type", type_string.c_str(), type_string.size());
	return result;
}

static yyjson_mut_val *LiteralToJson(yyjson_mut_doc *doc, const BoundConstantExpression &constant) {
	auto result = yyjson_mut_obj(doc);
	auto type_string = constant.value.type().ToString();
	yyjson_mut_obj_add_str(doc, result, "kind", "literal");
	yyjson_mut_obj_add_val(doc, result, "value", DuckValueToYyjsonLiteral(doc, constant.value));
	yyjson_mut_obj_add_strncpy(doc, result, "duckdb_type", type_string.c_str(), type_string.size());
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

static bool IsSupportedPredicateFunction(const string &name) {
	auto lower_name = StringUtil::Lower(name);
	return lower_name == "starts_with" || lower_name == "prefix" || lower_name == "contains" ||
	       lower_name == "contains_substr" || lower_name == "like" || lower_name == "~~";
}

static yyjson_mut_val *ExpressionToPredicate(yyjson_mut_doc *doc, const LogicalGet &get, const Expression &expr);

static yyjson_mut_val *ArithmeticFunctionToJson(yyjson_mut_doc *doc, const LogicalGet &get,
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
	auto args = yyjson_mut_arr(doc);
	for (auto &child : function.children) {
		yyjson_mut_arr_add_val(args, ExpressionToPredicate(doc, get, *child));
	}
	auto result = yyjson_mut_obj(doc);
	auto return_type = function.return_type.ToString();
	yyjson_mut_obj_add_str(doc, result, "kind", "arithmetic");
	yyjson_mut_obj_add_strncpy(doc, result, "op", function.function.name.c_str(), function.function.name.size());
	yyjson_mut_obj_add_strncpy(doc, result, "duckdb_type", return_type.c_str(), return_type.size());
	yyjson_mut_obj_add_val(doc, result, "args", args);
	return result;
}

static yyjson_mut_val *ScalarFunctionToPredicateJson(yyjson_mut_doc *doc, const LogicalGet &get,
                                                     const BoundFunctionExpression &function) {
	if (!IsSupportedPredicateFunction(function.function.name)) {
		throw NotImplementedException("LLM adapter cannot push down scalar function \"%s\" yet", function.function.name);
	}
	auto args = yyjson_mut_arr(doc);
	for (auto &child : function.children) {
		yyjson_mut_arr_add_val(args, ExpressionToPredicate(doc, get, *child));
	}
	auto result = yyjson_mut_obj(doc);
	auto return_type = function.return_type.ToString();
	yyjson_mut_obj_add_str(doc, result, "kind", "function");
	yyjson_mut_obj_add_strncpy(doc, result, "name", function.function.name.c_str(), function.function.name.size());
	yyjson_mut_obj_add_strncpy(doc, result, "duckdb_type", return_type.c_str(), return_type.size());
	yyjson_mut_obj_add_val(doc, result, "args", args);
	return result;
}

static yyjson_mut_val *ExpressionToPredicate(yyjson_mut_doc *doc, const LogicalGet &get, const Expression &expr) {
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_COLUMN_REF:
		return ColumnReferenceToJson(doc, get, expr.Cast<BoundColumnRefExpression>());
	case ExpressionClass::BOUND_REF:
		return BoundReferenceToJson(doc, get, expr.Cast<BoundReferenceExpression>());
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
		auto result = yyjson_mut_obj(doc);
		auto op = ComparisonOperatorToString(expr.GetExpressionType());
		yyjson_mut_obj_add_str(doc, result, "kind", "comparison");
		yyjson_mut_obj_add_strncpy(doc, result, "op", op.c_str(), op.size());
		yyjson_mut_obj_add_val(doc, result, "left", ExpressionToPredicate(doc, get, *comparison.left));
		yyjson_mut_obj_add_val(doc, result, "right", ExpressionToPredicate(doc, get, *comparison.right));
		return result;
	}
	case ExpressionClass::BOUND_CONJUNCTION: {
		auto &conjunction = expr.Cast<BoundConjunctionExpression>();
		auto children = yyjson_mut_arr(doc);
		for (auto &child : conjunction.children) {
			yyjson_mut_arr_add_val(children, ExpressionToPredicate(doc, get, *child));
		}
		auto result = yyjson_mut_obj(doc);
		yyjson_mut_obj_add_str(doc, result, "kind",
		                       expr.GetExpressionType() == ExpressionType::CONJUNCTION_AND ? "and" : "or");
		yyjson_mut_obj_add_val(doc, result, "children", children);
		return result;
	}
	case ExpressionClass::BOUND_FUNCTION: {
		auto &function = expr.Cast<BoundFunctionExpression>();
		if (IsSupportedArithmeticFunction(function.function.name)) {
			return ArithmeticFunctionToJson(doc, get, function);
		}
		return ScalarFunctionToPredicateJson(doc, get, function);
	}
	case ExpressionClass::BOUND_OPERATOR: {
		auto &op = expr.Cast<BoundOperatorExpression>();
		if (expr.GetExpressionType() == ExpressionType::OPERATOR_IS_NULL ||
		    expr.GetExpressionType() == ExpressionType::OPERATOR_IS_NOT_NULL) {
			if (op.children.size() != 1) {
				throw InternalException("Unexpected IS NULL child count");
			}
			auto result = yyjson_mut_obj(doc);
			yyjson_mut_obj_add_str(doc, result, "kind",
			                       expr.GetExpressionType() == ExpressionType::OPERATOR_IS_NULL ? "is_null" : "is_not_null");
			yyjson_mut_obj_add_val(doc, result, "expr", ExpressionToPredicate(doc, get, *op.children[0]));
			return result;
		}
		throw NotImplementedException("LLM adapter cannot push down operator \"%s\" yet",
		                              ExpressionTypeToString(expr.GetExpressionType()));
	}
	default:
		throw NotImplementedException("LLM adapter cannot push down predicate expression \"%s\" yet", expr.ToString());
	}
}

static string WriteJsonValue(yyjson_val *value) {
	size_t len = 0;
	yyjson_write_err error;
	auto data = yyjson_val_write_opts(value, YYJSON_WRITE_NOFLAG, nullptr, &len, &error);
	if (!data) {
		throw IOException("Failed to serialize LLM adapter streamed JSON: %s", error.msg ? error.msg : "unknown error");
	}
	string result(data, len);
	std::free(data);
	return result;
}

static bool StderrIsTerminal() {
#ifdef _WIN32
	return false;
#else
	return isatty(fileno(stderr)) != 0;
#endif
}

static bool ReadJsonNumber(yyjson_val *object, const char *key, double &result) {
	auto value = yyjson_obj_get(object, key);
	if (yyjson_is_int(value)) {
		result = static_cast<double>(yyjson_get_sint(value));
		return true;
	}
	if (yyjson_is_real(value)) {
		result = yyjson_get_real(value);
		return true;
	}
	return false;
}

static string ReadJsonString(yyjson_val *object, const char *key) {
	auto value = yyjson_obj_get(object, key);
	if (!yyjson_is_str(value)) {
		return string();
	}
	return string(yyjson_get_str(value), yyjson_get_len(value));
}

class LlmMutationProgressDisplay {
public:
	LlmMutationProgressDisplay() : enabled(StderrIsTerminal()) {
	}
	void Update(yyjson_val *event) {
		if (!enabled) {
			return;
		}
		double percent = -1.0;
		if (!ReadJsonNumber(event, "percent", percent)) {
			double step = 0;
			double total = 0;
			if (ReadJsonNumber(event, "step", step) && ReadJsonNumber(event, "total", total) && total > 0) {
				percent = 100.0 * step / total;
			}
		}
		if (percent < 0) {
			return;
		}
		if (percent > 100.0) {
			percent = 100.0;
		}
		auto phase = ReadJsonString(event, "phase");
		auto message = ReadJsonString(event, "message");
		Render(static_cast<int32_t>(percent + 0.5), phase, message);
	}
	void Finish() {
		if (!enabled || !printed) {
			return;
		}
		Render(100, "done", "mutation complete");
		std::fprintf(stderr, "\n");
		std::fflush(stderr);
	}

private:
	void Render(int32_t percent, const string &phase, const string &message) {
		percent = MaxValue<int32_t>(0, MinValue<int32_t>(100, percent));
		if (printed && percent == last_percent && phase == last_phase && message == last_message) {
			return;
		}
		ProgressBarDisplayInfo display_info;
		display_info.width = 28;
		auto bar = TerminalProgressBarDisplay::FormatProgressBar(display_info, percent);
		auto shown_phase = phase.empty() ? "working" : phase;
		auto shown_message = message.size() > 72 ? message.substr(0, 72) : message;
		// \033[K clears from the cursor to the end of the line so a short message after
		// a long one does not leave stale characters.
		std::fprintf(stderr, "\rLLM mutation %-10s %s %3d%% %s\033[K", shown_phase.c_str(), bar.c_str(), percent,
		             shown_message.c_str());
		std::fflush(stderr);
		printed = true;
		last_percent = percent;
		last_phase = phase;
		last_message = message;
	}
	bool enabled;
	bool printed = false;
	int32_t last_percent = -1;
	string last_phase;
	string last_message;
};

static bool HandleMutationStreamLine(string line, LlmMutationProgressDisplay &progress, string &response_json,
                                     string &error_message) {
	yyjson_read_err error;
	auto doc = yyjson_read_opts(line.empty() ? nullptr : &line[0], line.size(), YYJSON_READ_NOFLAG, nullptr, &error);
	if (!doc) {
		error_message = StringUtil::Format("malformed mutation progress JSON at byte %llu: %s", error.pos,
		                                   error.msg ? error.msg : "unknown error");
		return false;
	}
	unique_ptr<yyjson_doc, void (*)(yyjson_doc *)> guard(doc, yyjson_doc_free);
	auto root = yyjson_doc_get_root(doc);
	if (!yyjson_is_obj(root)) {
		return true;
	}
	if (auto response = yyjson_obj_get(root, "response")) {
		response_json = WriteJsonValue(response);
		return true;
	}
	auto status_json = yyjson_obj_get(root, "status");
	auto catalog_json = yyjson_obj_get(root, "catalog");
	if (yyjson_is_str(status_json) && catalog_json) {
		response_json = line;
		return true;
	}
	auto event = ReadJsonString(root, "event");
	if (event == "mutation_error") {
		if (auto detail = yyjson_obj_get(root, "detail")) {
			error_message = yyjson_is_str(detail) ? string(yyjson_get_str(detail), yyjson_get_len(detail))
			                                      : WriteJsonValue(detail);
		} else {
			error_message = "mutation stream ended with an error";
		}
		return true;
	}
	progress.Update(root);
	return true;
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
	string Post(ClientContext &, const string &path, const string &body) const {
		return HttpPostJson(parsed_endpoint, path, body);
	}
	void PostLines(ClientContext &, const string &path, const string &body,
	               const std::function<bool(const string &)> &line_callback) const {
		HttpPostJsonLines(parsed_endpoint, path, body, line_callback);
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

static BindInfo LlmScanBindInfo(const optional_ptr<FunctionData> bind_data);

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

static BindInfo LlmScanBindInfo(const optional_ptr<FunctionData> bind_data) {
	auto &bind = bind_data->Cast<LlmScanBindData>();
	return BindInfo(static_cast<TableCatalogEntry &>(bind.table));
}

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
	                             optional_ptr<PhysicalOperator> plan) override;
	PhysicalOperator &PlanDelete(ClientContext &context, PhysicalPlanGenerator &planner, LogicalDelete &op,
	                             PhysicalOperator &plan) override {
		throw NotImplementedException("LLM DELETE is not implemented in the adapter protocol yet");
	}
	PhysicalOperator &PlanUpdate(ClientContext &context, PhysicalPlanGenerator &planner, LogicalUpdate &op) override;
	PhysicalOperator &PlanUpdate(ClientContext &context, PhysicalPlanGenerator &planner, LogicalUpdate &op,
	                             PhysicalOperator &plan) override;
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
		auto doc = yyjson_mut_doc_new(nullptr);
		if (!doc) {
			throw InternalException("Failed to allocate LLM adapter JSON document");
		}
		auto request = yyjson_mut_obj(doc);
		yyjson_mut_doc_set_root(doc, request);
		auto catalog_name = GetName();
		yyjson_mut_obj_add_str(doc, request, "type", "introspect_catalog");
		yyjson_mut_obj_add_strncpy(doc, request, "catalog", catalog_name.c_str(), catalog_name.size());
		yyjson_mut_obj_add_strncpy(doc, request, "checkpoint_ref", client.CheckpointRef().c_str(),
		                           client.CheckpointRef().size());
		auto response = client.Post(context, "/v1/catalog/introspect", WriteJsonAndFree(doc));
		yyjson_read_err error;
		auto response_doc = yyjson_read_opts(response.empty() ? nullptr : &response[0], response.size(), YYJSON_READ_NOFLAG,
		                                     nullptr, &error);
		if (!response_doc) {
			throw InvalidInputException("Malformed LLM adapter JSON in catalog introspection at byte %llu: %s", error.pos,
			                            error.msg ? error.msg : "unknown error");
		}
		unique_ptr<yyjson_doc, void (*)(yyjson_doc *)> response_guard(response_doc, yyjson_doc_free);
		auto root = yyjson_doc_get_root(response_doc);
		if (!root) {
			throw InvalidInputException("Malformed LLM adapter JSON in catalog introspection: missing root value");
		}
		return ParseCatalogSnapshot(context, root);
	}
	void ReplaceCatalog(ClientContext &context, const LlmCatalogSnapshot &snapshot) {
		catalog_version = snapshot.version;
		main_schema->ReplaceTables(context, snapshot.tables);
	}
	optional_ptr<CatalogEntry> ApplyCreateTable(CatalogTransaction transaction, BoundCreateTableInfo &info);
	struct MutationResult {
		LlmCatalogSnapshot snapshot;
		idx_t affected_rows = 0;
	};
	MutationResult ApplyMutation(ClientContext &context, yyjson_mut_doc *doc, yyjson_mut_val *operations);
	string Select(ClientContext &context, const string &query_json) {
		auto doc = yyjson_mut_doc_new(nullptr);
		if (!doc) {
			throw InternalException("Failed to allocate LLM adapter JSON document");
		}
		auto request = yyjson_mut_obj(doc);
		yyjson_mut_doc_set_root(doc, request);
		yyjson_mut_obj_add_str(doc, request, "type", "select");
		yyjson_mut_obj_add_strncpy(doc, request, "catalog_version", catalog_version.c_str(), catalog_version.size());
		yyjson_mut_obj_add_val(doc, request, "query", yyjson_mut_rawncpy(doc, query_json.c_str(), query_json.size()));
		return client.Post(context, "/v1/query/select", WriteJsonAndFree(doc));
	}

private:
	void DropSchema(ClientContext &context, DropInfo &info) override {
		throw BinderException("LLM catalog does not support dropping schemas");
	}
	LlmCatalogSnapshot ParseCatalogSnapshot(ClientContext &context, yyjson_val *json) {
		LlmCatalogSnapshot snapshot;
		if (!yyjson_is_obj(json)) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected object for catalog snapshot");
		}
		auto version = yyjson_obj_get(json, "catalog_version");
		if (!yyjson_is_str(version)) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected string for catalog_version");
		}
		snapshot.version = string(yyjson_get_str(version), yyjson_get_len(version));
		auto schemas = yyjson_obj_get(json, "schemas");
		if (!yyjson_is_arr(schemas)) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected array for catalog snapshot schemas");
		}
		size_t schema_idx, schema_max;
		yyjson_val *schema_json;
		yyjson_arr_foreach(schemas, schema_idx, schema_max, schema_json) {
			if (!yyjson_is_obj(schema_json)) {
				throw InvalidInputException("Malformed LLM adapter JSON: expected object for schema");
			}
			auto schema_name_json = yyjson_obj_get(schema_json, "name");
			if (!yyjson_is_str(schema_name_json)) {
				throw InvalidInputException("Malformed LLM adapter JSON: expected string for schema.name");
			}
			auto schema_name = string(yyjson_get_str(schema_name_json), yyjson_get_len(schema_name_json));
			if (schema_name != DEFAULT_SCHEMA) {
				throw NotImplementedException("LLM adapter only supports the \"%s\" schema for now",
				                              string(DEFAULT_SCHEMA));
			}
			auto tables = yyjson_obj_get(schema_json, "tables");
			if (!yyjson_is_arr(tables)) {
				throw InvalidInputException("Malformed LLM adapter JSON: expected array for schema.tables");
			}
			size_t table_idx, table_max;
			yyjson_val *table_json;
			yyjson_arr_foreach(tables, table_idx, table_max, table_json) {
				if (!yyjson_is_obj(table_json)) {
					throw InvalidInputException("Malformed LLM adapter JSON: expected object for table");
				}
				LlmTableMeta table;
				table.schema = schema_name;
				auto table_name = yyjson_obj_get(table_json, "name");
				if (!yyjson_is_str(table_name)) {
					throw InvalidInputException("Malformed LLM adapter JSON: expected string for table.name");
				}
				table.name = string(yyjson_get_str(table_name), yyjson_get_len(table_name));
				auto columns = yyjson_obj_get(table_json, "columns");
				if (!yyjson_is_arr(columns)) {
					throw InvalidInputException("Malformed LLM adapter JSON: expected array for table.columns");
				}
				size_t column_idx, column_max;
				yyjson_val *column_json;
				yyjson_arr_foreach(columns, column_idx, column_max, column_json) {
					if (!yyjson_is_obj(column_json)) {
						throw InvalidInputException("Malformed LLM adapter JSON: expected object for column");
					}
					LlmColumnMeta column;
					auto column_name = yyjson_obj_get(column_json, "name");
					if (!yyjson_is_str(column_name)) {
						throw InvalidInputException("Malformed LLM adapter JSON: expected string for column.name");
					}
					column.name = string(yyjson_get_str(column_name), yyjson_get_len(column_name));
					auto duckdb_type = yyjson_obj_get(column_json, "duckdb_type");
					if (!yyjson_is_str(duckdb_type)) {
						throw InvalidInputException("Malformed LLM adapter JSON: expected string for column.duckdb_type");
					}
					auto type_string = string(yyjson_get_str(duckdb_type), yyjson_get_len(duckdb_type));
					column.type = ParseAdapterType(context, type_string);
					auto nullable = yyjson_obj_get(column_json, "nullable");
					if (!yyjson_is_bool(nullable)) {
						throw InvalidInputException("Malformed LLM adapter JSON: expected boolean for column.nullable");
					}
					column.nullable = yyjson_get_bool(nullable);
					table.columns.push_back(std::move(column));
				}
				if (auto primary_key = yyjson_obj_get(table_json, "primary_key")) {
					if (!yyjson_is_arr(primary_key)) {
						throw InvalidInputException("Malformed LLM adapter JSON: expected array for table.primary_key");
					}
					size_t key_idx, key_max;
					yyjson_val *key_json;
					yyjson_arr_foreach(primary_key, key_idx, key_max, key_json) {
						if (!yyjson_is_str(key_json)) {
							throw InvalidInputException("Malformed LLM adapter JSON: expected string for primary_key column");
						}
						table.primary_key.push_back(string(yyjson_get_str(key_json), yyjson_get_len(key_json)));
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

static yyjson_mut_val *InsertRowsOpToJson(yyjson_mut_doc *doc, const string &catalog_name, const LlmTableEntry &table,
                                          const vector<vector<Value>> &rows);

class LlmInsertGlobalState : public GlobalSinkState {
public:
	mutex lock;
	vector<vector<Value>> rows;
	idx_t inserted_rows = 0;
};

class LlmInsertSourceState : public GlobalSourceState {
public:
	bool emitted = false;
};

class LlmPhysicalInsert : public PhysicalOperator {
public:
	LlmPhysicalInsert(PhysicalPlan &physical_plan, vector<LogicalType> types, LlmCatalog &catalog_p,
	                  LlmTableEntry &table_p, idx_t estimated_cardinality)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::INSERT, std::move(types), estimated_cardinality),
	      catalog(catalog_p), table(table_p) {
	}

	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &) const override {
		return make_uniq<LlmInsertGlobalState>();
	}
	SinkResultType Sink(ExecutionContext &, DataChunk &chunk, OperatorSinkInput &input) const override {
		auto &state = input.global_state.Cast<LlmInsertGlobalState>();
		chunk.Flatten();
		lock_guard<mutex> guard(state.lock);
		for (idx_t row_idx = 0; row_idx < chunk.size(); row_idx++) {
			vector<Value> row;
			for (idx_t column_idx = 0; column_idx < chunk.ColumnCount(); column_idx++) {
				row.push_back(chunk.GetValue(column_idx, row_idx));
			}
			state.rows.push_back(std::move(row));
		}
		state.inserted_rows += chunk.size();
		return SinkResultType::NEED_MORE_INPUT;
	}
	SinkFinalizeType Finalize(Pipeline &, Event &, ClientContext &context, OperatorSinkFinalizeInput &input) const override {
		auto &state = input.global_state.Cast<LlmInsertGlobalState>();
		auto doc = yyjson_mut_doc_new(nullptr);
		if (!doc) {
			throw InternalException("Failed to allocate LLM adapter JSON document");
		}
		auto operations = yyjson_mut_arr(doc);
		yyjson_mut_arr_add_val(operations, InsertRowsOpToJson(doc, catalog.GetName(), table, state.rows));
		catalog.ApplyMutation(context, doc, operations);
		return SinkFinalizeType::READY;
	}
	bool IsSink() const override {
		return true;
	}
	bool SinkOrderDependent() const override {
		return true;
	}
	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &) const override {
		return make_uniq<LlmInsertSourceState>();
	}
	SourceResultType GetDataInternal(ExecutionContext &, DataChunk &chunk, OperatorSourceInput &input) const override {
		auto &source_state = input.global_state.Cast<LlmInsertSourceState>();
		if (source_state.emitted) {
			return SourceResultType::FINISHED;
		}
		auto &sink = sink_state->Cast<LlmInsertGlobalState>();
		chunk.SetCardinality(1);
		chunk.SetValue(0, 0, Value::BIGINT(NumericCast<int64_t>(sink.inserted_rows)));
		source_state.emitted = true;
		return SourceResultType::FINISHED;
	}
	bool IsSource() const override {
		return true;
	}

private:
	LlmCatalog &catalog;
	LlmTableEntry &table;
};

class LlmUpdateSourceState : public GlobalSourceState {
public:
	bool emitted = false;
};

class LlmPhysicalUpdate : public PhysicalOperator {
public:
	LlmPhysicalUpdate(PhysicalPlan &physical_plan, vector<LogicalType> types, LlmCatalog &catalog_p,
	                  string operation_json_p, idx_t estimated_cardinality)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::UPDATE, std::move(types), estimated_cardinality),
	      catalog(catalog_p), operation_json(std::move(operation_json_p)) {
	}

	unique_ptr<GlobalSourceState> GetGlobalSourceState(ClientContext &) const override {
		return make_uniq<LlmUpdateSourceState>();
	}
	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk, OperatorSourceInput &input) const override {
		auto &source_state = input.global_state.Cast<LlmUpdateSourceState>();
		if (source_state.emitted) {
			return SourceResultType::FINISHED;
		}
		auto doc = yyjson_mut_doc_new(nullptr);
		if (!doc) {
			throw InternalException("Failed to allocate LLM adapter JSON document");
		}
		auto operations = yyjson_mut_arr(doc);
		yyjson_mut_arr_add_val(operations, yyjson_mut_rawncpy(doc, operation_json.c_str(), operation_json.size()));
		auto result = catalog.ApplyMutation(context.client, doc, operations);
		chunk.SetCardinality(1);
		chunk.SetValue(0, 0, Value::BIGINT(NumericCast<int64_t>(result.affected_rows)));
		source_state.emitted = true;
		return SourceResultType::FINISHED;
	}
	bool IsSource() const override {
		return true;
	}

private:
	LlmCatalog &catalog;
	string operation_json;
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

static yyjson_mut_val *CreateTableOpToJson(yyjson_mut_doc *doc, const string &catalog_name,
                                           BoundCreateTableInfo &info) {
	auto &base = info.Base();
	ValidateCreateTable(base);
	auto columns = yyjson_mut_arr(doc);
	idx_t column_idx = 0;
	for (auto &column : base.columns.Logical()) {
		auto column_json = yyjson_mut_obj(doc);
		auto column_name = column.Name();
		auto column_type = column.Type().ToString();
		yyjson_mut_obj_add_strncpy(doc, column_json, "name", column_name.c_str(), column_name.size());
		yyjson_mut_obj_add_strncpy(doc, column_json, "duckdb_type", column_type.c_str(), column_type.size());
		yyjson_mut_obj_add_bool(doc, column_json, "nullable", ColumnIsNullable(base, column_idx));
		yyjson_mut_obj_add_null(doc, column_json, "default");
		yyjson_mut_obj_add_bool(doc, column_json, "generated", false);
		yyjson_mut_arr_add_val(columns, column_json);
		column_idx++;
	}
	auto primary_key = yyjson_mut_arr(doc);
	for (auto &column : ExtractPrimaryKey(base)) {
		yyjson_mut_arr_add_val(primary_key, yyjson_mut_strncpy(doc, column.c_str(), column.size()));
	}
	auto result = yyjson_mut_obj(doc);
	yyjson_mut_obj_add_str(doc, result, "op", "create_table");
	yyjson_mut_obj_add_strncpy(doc, result, "catalog", catalog_name.c_str(), catalog_name.size());
	yyjson_mut_obj_add_strncpy(doc, result, "schema", base.schema.c_str(), base.schema.size());
	yyjson_mut_obj_add_strncpy(doc, result, "table", base.table.c_str(), base.table.size());
	yyjson_mut_obj_add_str(doc, result, "on_conflict", "error");
	yyjson_mut_obj_add_val(doc, result, "columns", columns);
	yyjson_mut_obj_add_val(doc, result, "primary_key", primary_key);
	yyjson_mut_obj_add_val(doc, result, "unique", yyjson_mut_arr(doc));
	yyjson_mut_obj_add_val(doc, result, "checks", yyjson_mut_arr(doc));
	yyjson_mut_obj_add_val(doc, result, "foreign_keys", yyjson_mut_arr(doc));
	return result;
}

static yyjson_mut_val *MutationColumnToJson(yyjson_mut_doc *doc, const LlmColumnMeta &column) {
	auto column_json = yyjson_mut_obj(doc);
	auto column_type = column.type.ToString();
	yyjson_mut_obj_add_strncpy(doc, column_json, "name", column.name.c_str(), column.name.size());
	yyjson_mut_obj_add_strncpy(doc, column_json, "duckdb_type", column_type.c_str(), column_type.size());
	yyjson_mut_obj_add_bool(doc, column_json, "nullable", column.nullable);
	return column_json;
}

static yyjson_mut_val *PrimaryKeyToJson(yyjson_mut_doc *doc, const vector<string> &primary_key_columns) {
	auto primary_key = yyjson_mut_arr(doc);
	for (auto &column : primary_key_columns) {
		yyjson_mut_arr_add_val(primary_key, yyjson_mut_strncpy(doc, column.c_str(), column.size()));
	}
	return primary_key;
}

static yyjson_mut_val *InsertRowsOpToJson(yyjson_mut_doc *doc, const string &catalog_name, const LlmTableEntry &table,
                                          const vector<vector<Value>> &rows) {
	auto &meta = table.GetMeta();
	auto columns = yyjson_mut_arr(doc);
	for (auto &column : meta.columns) {
		yyjson_mut_arr_add_val(columns, MutationColumnToJson(doc, column));
	}
	auto rows_json = yyjson_mut_arr(doc);
	for (auto &row : rows) {
		if (row.size() != meta.columns.size()) {
			throw InternalException("LLM insert row width does not match table width");
		}
		auto row_json = yyjson_mut_arr(doc);
		for (idx_t column_idx = 0; column_idx < row.size(); column_idx++) {
			yyjson_mut_arr_add_val(row_json, DuckValueToYyjsonLiteral(doc, row[column_idx]));
		}
		yyjson_mut_arr_add_val(rows_json, row_json);
	}
	auto result = yyjson_mut_obj(doc);
	yyjson_mut_obj_add_str(doc, result, "op", "insert_rows");
	yyjson_mut_obj_add_strncpy(doc, result, "catalog", catalog_name.c_str(), catalog_name.size());
	yyjson_mut_obj_add_strncpy(doc, result, "schema", meta.schema.c_str(), meta.schema.size());
	yyjson_mut_obj_add_strncpy(doc, result, "table", meta.name.c_str(), meta.name.size());
	yyjson_mut_obj_add_val(doc, result, "columns", columns);
	yyjson_mut_obj_add_val(doc, result, "primary_key", PrimaryKeyToJson(doc, meta.primary_key));
	yyjson_mut_obj_add_val(doc, result, "rows", rows_json);
	return result;
}

static void FindSingleLogicalGetRecursive(LogicalOperator &node, optional_ptr<LogicalGet> &result) {
	if (node.type == LogicalOperatorType::LOGICAL_GET) {
		if (result) {
			throw NotImplementedException("LLM UPDATE only supports a single target table scan");
		}
		result = &node.Cast<LogicalGet>();
	}
	for (auto &child : node.children) {
		FindSingleLogicalGetRecursive(*child, result);
	}
}

static optional_ptr<LogicalGet> FindSingleLogicalGet(LogicalOperator &op) {
	optional_ptr<LogicalGet> result;
	FindSingleLogicalGetRecursive(op, result);
	return result;
}

static void CollectFilterExpressions(LogicalOperator &op, vector<reference<Expression>> &filters) {
	if (op.type == LogicalOperatorType::LOGICAL_FILTER) {
		auto &filter = op.Cast<LogicalFilter>();
		for (auto &expr : filter.expressions) {
			filters.push_back(*expr);
		}
	}
	for (auto &child : op.children) {
		CollectFilterExpressions(*child, filters);
	}
}

static yyjson_mut_val *BuildUpdatePredicateJson(yyjson_mut_doc *doc, const LogicalGet &get, LogicalOperator &root) {
	vector<reference<Expression>> filters;
	CollectFilterExpressions(root, filters);
	if (filters.empty()) {
		if (get.bind_data) {
			auto &bind = get.bind_data->Cast<LlmScanBindData>();
			if (bind.has_predicate) {
				return yyjson_mut_rawncpy(doc, bind.predicate_json.c_str(), bind.predicate_json.size());
			}
		}
		return yyjson_mut_null(doc);
	}
	if (filters.size() == 1) {
		return ExpressionToPredicate(doc, get, filters[0].get());
	}
	auto children = yyjson_mut_arr(doc);
	for (auto &filter : filters) {
		yyjson_mut_arr_add_val(children, ExpressionToPredicate(doc, get, filter.get()));
	}
	auto result = yyjson_mut_obj(doc);
	yyjson_mut_obj_add_str(doc, result, "kind", "and");
	yyjson_mut_obj_add_val(doc, result, "children", children);
	return result;
}

static const LogicalProjection &UpdateProjection(LogicalUpdate &op) {
	if (op.children.empty() || op.children[0]->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		throw NotImplementedException("LLM UPDATE only supports direct update projections");
	}
	return op.children[0]->Cast<LogicalProjection>();
}

static yyjson_mut_val *UpdateRowsOpToJson(yyjson_mut_doc *doc, const string &catalog_name, const LlmTableEntry &table,
                                          LogicalUpdate &op) {
	auto &meta = table.GetMeta();
	auto columns = yyjson_mut_arr(doc);
	for (auto &column : meta.columns) {
		yyjson_mut_arr_add_val(columns, MutationColumnToJson(doc, column));
	}
	auto logical_get = FindSingleLogicalGet(*op.children[0]);
	if (!logical_get) {
		throw NotImplementedException("LLM UPDATE could not find the target table scan");
	}
	auto &projection = UpdateProjection(op);
	if (projection.expressions.size() < op.columns.size()) {
		throw InternalException("LLM UPDATE projection does not contain all assignment expressions");
	}
	auto assignments = yyjson_mut_arr(doc);
	for (idx_t assignment_idx = 0; assignment_idx < op.columns.size(); assignment_idx++) {
		auto column_idx = op.columns[assignment_idx].index;
		if (column_idx >= meta.columns.size()) {
			throw NotImplementedException("LLM UPDATE cannot update virtual columns");
		}
		auto &column = meta.columns[column_idx];
		auto assignment = yyjson_mut_obj(doc);
		auto column_type = column.type.ToString();
		yyjson_mut_obj_add_strncpy(doc, assignment, "column", column.name.c_str(), column.name.size());
		yyjson_mut_obj_add_strncpy(doc, assignment, "duckdb_type", column_type.c_str(), column_type.size());
		yyjson_mut_obj_add_val(doc, assignment, "value",
		                       ExpressionToPredicate(doc, *logical_get, *projection.expressions[assignment_idx]));
		yyjson_mut_arr_add_val(assignments, assignment);
	}
	auto result = yyjson_mut_obj(doc);
	yyjson_mut_obj_add_str(doc, result, "op", "update_rows");
	yyjson_mut_obj_add_strncpy(doc, result, "catalog", catalog_name.c_str(), catalog_name.size());
	yyjson_mut_obj_add_strncpy(doc, result, "schema", meta.schema.c_str(), meta.schema.size());
	yyjson_mut_obj_add_strncpy(doc, result, "table", meta.name.c_str(), meta.name.size());
	yyjson_mut_obj_add_val(doc, result, "columns", columns);
	yyjson_mut_obj_add_val(doc, result, "primary_key", PrimaryKeyToJson(doc, meta.primary_key));
	yyjson_mut_obj_add_val(doc, result, "assignments", assignments);
	yyjson_mut_obj_add_val(doc, result, "predicate", BuildUpdatePredicateJson(doc, *logical_get, *op.children[0]));
	return result;
}

static string BuildUpdateOperationJson(const string &catalog_name, const LlmTableEntry &table, LogicalUpdate &op) {
	auto doc = yyjson_mut_doc_new(nullptr);
	if (!doc) {
		throw InternalException("Failed to allocate LLM adapter JSON document");
	}
	auto operation = UpdateRowsOpToJson(doc, catalog_name, table, op);
	yyjson_mut_doc_set_root(doc, operation);
	return WriteJsonAndFree(doc);
}

LlmCatalog::MutationResult LlmCatalog::ApplyMutation(ClientContext &context, yyjson_mut_doc *doc,
                                                     yyjson_mut_val *operations) {
	auto request = yyjson_mut_obj(doc);
	yyjson_mut_doc_set_root(doc, request);
	yyjson_mut_obj_add_str(doc, request, "type", "apply_mutation");
	yyjson_mut_obj_add_strncpy(doc, request, "base_catalog_version", catalog_version.c_str(), catalog_version.size());
	yyjson_mut_obj_add_val(doc, request, "operations", operations);
	string response;
	string stream_error;
	LlmMutationProgressDisplay progress;
	client.PostLines(context, "/v1/mutations/apply", WriteJsonAndFree(doc),
	                 [&](const string &line) { return HandleMutationStreamLine(line, progress, response, stream_error); });
	progress.Finish();
	if (!stream_error.empty()) {
		throw IOException("LLM mutation failed: %s", stream_error);
	}
	if (response.empty()) {
		throw IOException("LLM mutation stream ended without a mutation response");
	}
	yyjson_read_err error;
	auto response_doc = yyjson_read_opts(response.empty() ? nullptr : &response[0], response.size(), YYJSON_READ_NOFLAG,
	                                     nullptr, &error);
	if (!response_doc) {
		throw InvalidInputException("Malformed LLM adapter JSON in mutation response at byte %llu: %s", error.pos,
		                            error.msg ? error.msg : "unknown error");
	}
	unique_ptr<yyjson_doc, void (*)(yyjson_doc *)> response_guard(response_doc, yyjson_doc_free);
	auto root = yyjson_doc_get_root(response_doc);
	if (!yyjson_is_obj(root)) {
		throw InvalidInputException("Malformed LLM adapter JSON: expected object for mutation response");
	}
	auto status_json = yyjson_obj_get(root, "status");
	if (!yyjson_is_str(status_json)) {
		throw InvalidInputException("Malformed LLM adapter JSON: expected string for mutation status");
	}
	auto status = string(yyjson_get_str(status_json), yyjson_get_len(status_json));
	if (status != "applied") {
		throw IOException("LLM mutation failed with status \"%s\"", status);
	}
	auto catalog = yyjson_obj_get(root, "catalog");
	if (!catalog) {
		throw InvalidInputException("Malformed LLM adapter JSON: missing catalog in mutation response");
	}
	MutationResult result;
	result.snapshot = ParseCatalogSnapshot(context, catalog);
	if (auto metrics = yyjson_obj_get(root, "metrics")) {
		if (!yyjson_is_obj(metrics)) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected object for mutation metrics");
		}
		if (auto affected = yyjson_obj_get(metrics, "affected_rows")) {
			if (yyjson_is_int(affected)) {
				result.affected_rows = NumericCast<idx_t>(yyjson_get_sint(affected));
			}
		}
	}
	ReplaceCatalog(context, result.snapshot);
	return result;
}

optional_ptr<CatalogEntry> LlmCatalog::ApplyCreateTable(CatalogTransaction transaction, BoundCreateTableInfo &info) {
	if (!transaction.HasContext()) {
		throw InternalException("LLM CREATE TABLE requires a client context");
	}
	auto &context = transaction.GetContext();
	auto doc = yyjson_mut_doc_new(nullptr);
	if (!doc) {
		throw InternalException("Failed to allocate LLM adapter JSON document");
	}
	auto op = CreateTableOpToJson(doc, GetName(), info);
	auto operations = yyjson_mut_arr(doc);
	yyjson_mut_arr_add_val(operations, op);
	ApplyMutation(context, doc, operations);
	return main_schema->LookupEntry(transaction, EntryLookupInfo(CatalogType::TABLE_ENTRY, info.Base().table));
}

optional_ptr<CatalogEntry> LlmSchemaEntry::CreateTable(CatalogTransaction transaction, BoundCreateTableInfo &info) {
	return llm_catalog.ApplyCreateTable(transaction, info);
}

PhysicalOperator &LlmCatalog::PlanInsert(ClientContext &context, PhysicalPlanGenerator &planner, LogicalInsert &op,
                                         optional_ptr<PhysicalOperator> plan) {
	if (!plan) {
		throw NotImplementedException("LLM INSERT requires explicit input rows");
	}
	if (op.return_chunk) {
		throw NotImplementedException("LLM INSERT does not support RETURNING yet");
	}
	if (op.on_conflict_info.action_type != OnConflictAction::THROW) {
		throw NotImplementedException("LLM INSERT does not support ON CONFLICT yet");
	}
	if (!op.column_index_map.empty()) {
		plan = planner.ResolveDefaultsProjection(op, *plan);
	}
	auto &table = op.table.Cast<LlmTableEntry>();
	auto &insert = planner.Make<LlmPhysicalInsert>(std::move(op.types), *this, table, op.estimated_cardinality);
	insert.children.push_back(*plan);
	return insert;
}

PhysicalOperator &LlmCatalog::PlanUpdate(ClientContext &context, PhysicalPlanGenerator &planner, LogicalUpdate &op) {
	if (op.return_chunk) {
		throw NotImplementedException("LLM UPDATE does not support RETURNING yet");
	}
	auto &table = op.table.Cast<LlmTableEntry>();
	auto operation_json = BuildUpdateOperationJson(GetName(), table, op);
	return planner.Make<LlmPhysicalUpdate>(std::move(op.types), *this, std::move(operation_json),
	                                       op.estimated_cardinality);
}

PhysicalOperator &LlmCatalog::PlanUpdate(ClientContext &context, PhysicalPlanGenerator &planner, LogicalUpdate &op,
                                         PhysicalOperator &plan) {
	return PlanUpdate(context, planner, op);
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

static yyjson_mut_val *BuildProjectionJson(yyjson_mut_doc *doc, const LlmTableEntry &table,
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

	auto projections = yyjson_mut_arr(doc);
	for (auto column_id : response_column_ids) {
		auto &column = meta.columns[column_id];
		response_types.push_back(column.type);
		auto projection = yyjson_mut_obj(doc);
		auto type_string = column.type.ToString();
		yyjson_mut_obj_add_strncpy(doc, projection, "name", column.name.c_str(), column.name.size());
		yyjson_mut_obj_add_strncpy(doc, projection, "duckdb_type", type_string.c_str(), type_string.size());
		yyjson_mut_arr_add_val(projections, projection);
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
	auto doc = yyjson_mut_doc_new(nullptr);
	if (!doc) {
		throw InternalException("Failed to allocate LLM adapter JSON document");
	}
	auto projection = BuildProjectionJson(doc, bind.table, output_column_ids, result->response_types, result->output_types,
	                                      result->output_to_response);
	auto table_columns = yyjson_mut_arr(doc);
	for (auto &column : bind.table.GetMeta().columns) {
		yyjson_mut_arr_add_val(table_columns, MutationColumnToJson(doc, column));
	}
	auto query = yyjson_mut_obj(doc);
	yyjson_mut_doc_set_root(doc, query);
	yyjson_mut_obj_add_strncpy(doc, query, "schema", bind.table.GetMeta().schema.c_str(),
	                           bind.table.GetMeta().schema.size());
	yyjson_mut_obj_add_strncpy(doc, query, "table", bind.table.GetMeta().name.c_str(),
	                           bind.table.GetMeta().name.size());
	yyjson_mut_obj_add_val(doc, query, "columns", table_columns);
	yyjson_mut_obj_add_val(doc, query, "primary_key", PrimaryKeyToJson(doc, bind.table.GetMeta().primary_key));
	yyjson_mut_obj_add_val(doc, query, "projection", projection);
	yyjson_mut_obj_add_val(doc, query, "predicate",
	                       bind.has_predicate ? yyjson_mut_rawncpy(doc, bind.predicate_json.c_str(),
	                                                               bind.predicate_json.size())
	                                          : yyjson_mut_null(doc));
	if (bind.has_limit) {
		yyjson_mut_obj_add_int(doc, query, "limit", NumericCast<int64_t>(bind.limit));
	} else {
		yyjson_mut_obj_add_null(doc, query, "limit");
	}
	auto response = bind.catalog.Select(context, WriteJsonAndFree(doc));
	yyjson_read_err error;
	auto response_doc = yyjson_read_opts(response.empty() ? nullptr : &response[0], response.size(), YYJSON_READ_NOFLAG,
	                                     nullptr, &error);
	if (!response_doc) {
		throw InvalidInputException("Malformed LLM adapter JSON in select response at byte %llu: %s", error.pos,
		                            error.msg ? error.msg : "unknown error");
	}
	unique_ptr<yyjson_doc, void (*)(yyjson_doc *)> response_guard(response_doc, yyjson_doc_free);
	auto root = yyjson_doc_get_root(response_doc);
	if (!yyjson_is_obj(root)) {
		throw InvalidInputException("Malformed LLM adapter JSON: expected object for select response");
	}
	auto columns = yyjson_obj_get(root, "columns");
	if (!yyjson_is_arr(columns)) {
		throw InvalidInputException("Malformed LLM adapter JSON: expected array for select columns");
	}
	if (yyjson_arr_size(columns) != result->response_types.size()) {
		throw IOException("LLM adapter select returned %llu columns, expected %llu",
		                  NumericCast<idx_t>(yyjson_arr_size(columns)),
		                  result->response_types.size());
	}
	size_t column_idx, column_max;
	yyjson_val *column_json;
	yyjson_arr_foreach(columns, column_idx, column_max, column_json) {
		if (!yyjson_is_obj(column_json)) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected object for select column");
		}
		auto duckdb_type = yyjson_obj_get(column_json, "duckdb_type");
		if (!yyjson_is_str(duckdb_type)) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected string for select column type");
		}
		auto returned_type = string(yyjson_get_str(duckdb_type), yyjson_get_len(duckdb_type));
		if (returned_type != result->response_types[column_idx].ToString()) {
			throw IOException("LLM adapter select returned type \"%s\" for column %llu, expected \"%s\"", returned_type,
			                  column_idx, result->response_types[column_idx].ToString());
		}
	}
	auto rows = yyjson_obj_get(root, "rows");
	if (!yyjson_is_arr(rows)) {
		throw InvalidInputException("Malformed LLM adapter JSON: expected array for select rows");
	}
	size_t row_idx, row_max;
	yyjson_val *row_json;
	yyjson_arr_foreach(rows, row_idx, row_max, row_json) {
		if (!yyjson_is_arr(row_json)) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected array for select row");
		}
		if (yyjson_arr_size(row_json) != result->response_types.size()) {
			throw IOException("LLM adapter select returned a row with the wrong width");
		}
		vector<Value> row_values;
		size_t value_idx, value_max;
		yyjson_val *value_json;
		yyjson_arr_foreach(row_json, value_idx, value_max, value_json) {
			row_values.push_back(YyjsonLiteralToDuckValue(value_json, result->response_types[value_idx], "select row value"));
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
	auto doc = yyjson_mut_doc_new(nullptr);
	if (!doc) {
		throw InternalException("Failed to allocate LLM adapter JSON document");
	}
	auto predicates = yyjson_mut_arr(doc);
	for (auto &filter : filters) {
		yyjson_mut_arr_add_val(predicates, ExpressionToPredicate(doc, get, *filter));
	}
	bind.has_predicate = true;
	yyjson_mut_val *predicate = nullptr;
	if (yyjson_mut_arr_size(predicates) == 1) {
		predicate = yyjson_mut_arr_get(predicates, 0);
	} else {
		predicate = yyjson_mut_obj(doc);
		yyjson_mut_obj_add_str(doc, predicate, "kind", "and");
		yyjson_mut_obj_add_val(doc, predicate, "children", predicates);
	}
	yyjson_mut_doc_set_root(doc, predicate);
	bind.predicate_json = WriteJsonAndFree(doc);
	filters.clear();
}

TableFunction LlmTableEntry::GetScanFunction(ClientContext &, unique_ptr<FunctionData> &bind_data) {
	bind_data = make_uniq<LlmScanBindData>(llm_catalog, *this);
	TableFunction scan("llm_scan", {}, LlmScanFunction, nullptr, LlmScanInitGlobal, nullptr);
	scan.projection_pushdown = true;
	scan.filter_prune = true;
	scan.pushdown_complex_filter = LlmPushdownComplexFilter;
	scan.get_bind_info = LlmScanBindInfo;
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
