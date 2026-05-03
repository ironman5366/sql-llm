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
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/operator/logical_create_table.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_update.hpp"
#include "duckdb/planner/parsed_data/bound_create_table_info.hpp"
#include "duckdb/storage/database_size.hpp"
#include "duckdb/storage/storage_extension.hpp"
#include "duckdb/storage/table_storage_info.hpp"
#include "duckdb/transaction/transaction.hpp"
#include "duckdb/transaction/transaction_manager.hpp"

#include "httplib.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>

namespace duckdb {

class LlmCatalog;
class LlmSchemaEntry;
class LlmTableEntry;

class JsonValue {
public:
	enum class Type : uint8_t { NULL_VALUE, BOOLEAN, NUMBER, STRING, ARRAY, OBJECT };

	JsonValue() : type(Type::NULL_VALUE), boolean(false), number(0), integer(0), integral(true) {
	}
	static JsonValue Null() { return JsonValue(); }
	static JsonValue Boolean(bool value) {
		JsonValue result;
		result.type = Type::BOOLEAN;
		result.boolean = value;
		return result;
	}
	static JsonValue Number(int64_t value) {
		JsonValue result;
		result.type = Type::NUMBER;
		result.number = static_cast<double>(value);
		result.integer = value;
		result.integral = true;
		return result;
	}
	static JsonValue Number(double value) {
		JsonValue result;
		result.type = Type::NUMBER;
		result.number = value;
		result.integer = static_cast<int64_t>(value);
		result.integral = false;
		return result;
	}
	static JsonValue String(string value) {
		JsonValue result;
		result.type = Type::STRING;
		result.string_value = std::move(value);
		return result;
	}
	static JsonValue Array(vector<JsonValue> values) {
		JsonValue result;
		result.type = Type::ARRAY;
		result.array = std::move(values);
		return result;
	}
	static JsonValue Object(vector<std::pair<string, JsonValue>> values) {
		JsonValue result;
		result.type = Type::OBJECT;
		result.object = std::move(values);
		return result;
	}
	bool IsNull() const { return type == Type::NULL_VALUE; }
	const JsonValue *Get(const string &key) const {
		if (type != Type::OBJECT) {
			return nullptr;
		}
		for (auto &entry : object) {
			if (entry.first == key) {
				return &entry.second;
			}
		}
		return nullptr;
	}
	const JsonValue &Require(const string &key, const string &context) const {
		auto value = Get(key);
		if (!value) {
			throw InvalidInputException("Malformed LLM adapter JSON: missing field \"%s\" in %s", key, context);
		}
		return *value;
	}
	bool GetBoolean(const string &context) const {
		if (type != Type::BOOLEAN) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected boolean for %s", context);
		}
		return boolean;
	}
	int64_t GetInteger(const string &context) const {
		if (type != Type::NUMBER || !integral) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected integer for %s", context);
		}
		return integer;
	}
	double GetDouble(const string &context) const {
		if (type != Type::NUMBER) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected number for %s", context);
		}
		return number;
	}
	const string &GetString(const string &context) const {
		if (type != Type::STRING) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected string for %s", context);
		}
		return string_value;
	}
	const vector<JsonValue> &GetArray(const string &context) const {
		if (type != Type::ARRAY) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected array for %s", context);
		}
		return array;
	}

private:
	Type type;
	bool boolean;
	double number;
	int64_t integer;
	bool integral;
	string string_value;
	vector<JsonValue> array;
	vector<std::pair<string, JsonValue>> object;
	friend class JsonParser;
	friend string SerializeJson(const JsonValue &value);
};

class JsonParser {
public:
	explicit JsonParser(const string &input) : input(input), position(0) {
	}
	JsonValue Parse() {
		auto result = ParseValue();
		SkipWhitespace();
		if (position != input.size()) {
			throw InvalidInputException("Malformed LLM adapter JSON: trailing characters at byte %llu", position);
		}
		return result;
	}

private:
	const string &input;
	idx_t position;
	void SkipWhitespace() {
		while (position < input.size() && std::isspace(static_cast<unsigned char>(input[position]))) {
			position++;
		}
	}
	char Peek() {
		SkipWhitespace();
		if (position >= input.size()) {
			throw InvalidInputException("Malformed LLM adapter JSON: unexpected end of input");
		}
		return input[position];
	}
	bool Consume(char expected) {
		SkipWhitespace();
		if (position < input.size() && input[position] == expected) {
			position++;
			return true;
		}
		return false;
	}
	void Expect(char expected) {
		if (!Consume(expected)) {
			throw InvalidInputException("Malformed LLM adapter JSON: expected '%c' at byte %llu", expected, position);
		}
	}
	bool MatchLiteral(const char *literal) {
		SkipWhitespace();
		auto len = strlen(literal);
		if (position + len > input.size() || input.compare(position, len, literal) != 0) {
			return false;
		}
		position += len;
		return true;
	}
	JsonValue ParseValue() {
		auto c = Peek();
		if (c == 'n') {
			if (!MatchLiteral("null")) {
				throw InvalidInputException("Malformed LLM adapter JSON: invalid null literal");
			}
			return JsonValue::Null();
		}
		if (c == 't') {
			if (!MatchLiteral("true")) {
				throw InvalidInputException("Malformed LLM adapter JSON: invalid true literal");
			}
			return JsonValue::Boolean(true);
		}
		if (c == 'f') {
			if (!MatchLiteral("false")) {
				throw InvalidInputException("Malformed LLM adapter JSON: invalid false literal");
			}
			return JsonValue::Boolean(false);
		}
		if (c == '"') {
			return JsonValue::String(ParseString());
		}
		if (c == '[') {
			return ParseArray();
		}
		if (c == '{') {
			return ParseObject();
		}
		if (c == '-' || (c >= '0' && c <= '9')) {
			return ParseNumber();
		}
		throw InvalidInputException("Malformed LLM adapter JSON: unexpected character '%c' at byte %llu", c, position);
	}
	static void AppendUtf8(string &result, uint32_t codepoint) {
		if (codepoint <= 0x7F) {
			result.push_back(static_cast<char>(codepoint));
		} else if (codepoint <= 0x7FF) {
			result.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
			result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
		} else {
			result.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
			result.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
			result.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
		}
	}
	static uint32_t HexValue(char c) {
		if (c >= '0' && c <= '9') {
			return c - '0';
		}
		if (c >= 'a' && c <= 'f') {
			return 10 + c - 'a';
		}
		if (c >= 'A' && c <= 'F') {
			return 10 + c - 'A';
		}
		throw InvalidInputException("Malformed LLM adapter JSON: invalid unicode escape");
	}
	uint32_t ParseUnicodeEscape() {
		if (position + 4 > input.size()) {
			throw InvalidInputException("Malformed LLM adapter JSON: truncated unicode escape");
		}
		uint32_t codepoint = 0;
		for (idx_t i = 0; i < 4; i++) {
			codepoint = (codepoint << 4) | HexValue(input[position++]);
		}
		return codepoint;
	}
	string ParseString() {
		Expect('"');
		string result;
		while (position < input.size()) {
			auto c = input[position++];
			if (c == '"') {
				return result;
			}
			if (c != '\\') {
				result.push_back(c);
				continue;
			}
			if (position >= input.size()) {
				throw InvalidInputException("Malformed LLM adapter JSON: truncated string escape");
			}
			auto escaped = input[position++];
			switch (escaped) {
			case '"':
			case '\\':
			case '/':
				result.push_back(escaped);
				break;
			case 'b':
				result.push_back('\b');
				break;
			case 'f':
				result.push_back('\f');
				break;
			case 'n':
				result.push_back('\n');
				break;
			case 'r':
				result.push_back('\r');
				break;
			case 't':
				result.push_back('\t');
				break;
			case 'u':
				AppendUtf8(result, ParseUnicodeEscape());
				break;
			default:
				throw InvalidInputException("Malformed LLM adapter JSON: invalid string escape '\\%c'", escaped);
			}
		}
		throw InvalidInputException("Malformed LLM adapter JSON: unterminated string");
	}
	JsonValue ParseArray() {
		Expect('[');
		vector<JsonValue> values;
		if (Consume(']')) {
			return JsonValue::Array(std::move(values));
		}
		while (true) {
			values.push_back(ParseValue());
			if (Consume(']')) {
				return JsonValue::Array(std::move(values));
			}
			Expect(',');
		}
	}
	JsonValue ParseObject() {
		Expect('{');
		vector<std::pair<string, JsonValue>> values;
		if (Consume('}')) {
			return JsonValue::Object(std::move(values));
		}
		while (true) {
			auto key = ParseString();
			Expect(':');
			values.emplace_back(std::move(key), ParseValue());
			if (Consume('}')) {
				return JsonValue::Object(std::move(values));
			}
			Expect(',');
		}
	}
	JsonValue ParseNumber() {
		SkipWhitespace();
		auto start = position;
		if (input[position] == '-') {
			position++;
		}
		if (position >= input.size()) {
			throw InvalidInputException("Malformed LLM adapter JSON: truncated number");
		}
		if (input[position] == '0') {
			position++;
		} else if (input[position] >= '1' && input[position] <= '9') {
			while (position < input.size() && input[position] >= '0' && input[position] <= '9') {
				position++;
			}
		} else {
			throw InvalidInputException("Malformed LLM adapter JSON: invalid number at byte %llu", start);
		}
		bool integral = true;
		if (position < input.size() && input[position] == '.') {
			integral = false;
			position++;
			while (position < input.size() && input[position] >= '0' && input[position] <= '9') {
				position++;
			}
		}
		if (position < input.size() && (input[position] == 'e' || input[position] == 'E')) {
			integral = false;
			position++;
			if (position < input.size() && (input[position] == '+' || input[position] == '-')) {
				position++;
			}
			while (position < input.size() && input[position] >= '0' && input[position] <= '9') {
				position++;
			}
		}
		auto text = input.substr(start, position - start);
		if (integral) {
			return JsonValue::Number(static_cast<int64_t>(std::strtoll(text.c_str(), nullptr, 10)));
		}
		return JsonValue::Number(std::strtod(text.c_str(), nullptr));
	}
};

static void AppendJsonString(string &result, const string &value) {
	result.push_back('"');
	for (auto c : value) {
		switch (c) {
		case '"':
			result += "\\\"";
			break;
		case '\\':
			result += "\\\\";
			break;
		case '\n':
			result += "\\n";
			break;
		case '\r':
			result += "\\r";
			break;
		case '\t':
			result += "\\t";
			break;
		default:
			result.push_back(c);
		}
	}
	result.push_back('"');
}

string SerializeJson(const JsonValue &value) {
	switch (value.type) {
	case JsonValue::Type::NULL_VALUE:
		return "null";
	case JsonValue::Type::BOOLEAN:
		return value.boolean ? "true" : "false";
	case JsonValue::Type::NUMBER:
		return value.integral ? std::to_string(value.integer) : StringUtil::Format("%.17g", value.number);
	case JsonValue::Type::STRING: {
		string result;
		AppendJsonString(result, value.string_value);
		return result;
	}
	case JsonValue::Type::ARRAY: {
		string result = "[";
		for (idx_t i = 0; i < value.array.size(); i++) {
			if (i > 0) {
				result += ",";
			}
			result += SerializeJson(value.array[i]);
		}
		result += "]";
		return result;
	}
	case JsonValue::Type::OBJECT: {
		string result = "{";
		for (idx_t i = 0; i < value.object.size(); i++) {
			if (i > 0) {
				result += ",";
			}
			AppendJsonString(result, value.object[i].first);
			result += ":";
			result += SerializeJson(value.object[i].second);
		}
		result += "}";
		return result;
	}
	default:
		throw InternalException("Unknown JSON value type");
	}
}

static JsonValue ParseJson(const string &input) {
	return JsonParser(input).Parse();
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

static JsonValue ValueToJson(const Value &value) {
	if (value.IsNull()) {
		return JsonValue::Null();
	}
	switch (value.type().id()) {
	case LogicalTypeId::BOOLEAN:
		return JsonValue::Boolean(value.GetValue<bool>());
	case LogicalTypeId::TINYINT:
		return JsonValue::Number(static_cast<int64_t>(value.GetValue<int8_t>()));
	case LogicalTypeId::SMALLINT:
		return JsonValue::Number(static_cast<int64_t>(value.GetValue<int16_t>()));
	case LogicalTypeId::INTEGER:
		return JsonValue::Number(static_cast<int64_t>(value.GetValue<int32_t>()));
	case LogicalTypeId::BIGINT:
		return JsonValue::Number(value.GetValue<int64_t>());
	case LogicalTypeId::FLOAT:
		return JsonValue::Number(static_cast<double>(value.GetValue<float>()));
	case LogicalTypeId::DOUBLE:
		return JsonValue::Number(value.GetValue<double>());
	case LogicalTypeId::VARCHAR:
		return JsonValue::String(value.GetValue<string>());
	default:
		throw NotImplementedException("LLM adapter cannot serialize literal of type \"%s\" yet", value.type().ToString());
	}
}

static Value JsonToValue(const JsonValue &json, const LogicalType &type, const string &context) {
	if (json.IsNull()) {
		return Value(type);
	}
	switch (type.id()) {
	case LogicalTypeId::BOOLEAN:
		return Value::BOOLEAN(json.GetBoolean(context));
	case LogicalTypeId::TINYINT:
		return Value::TINYINT(NumericCast<int8_t>(json.GetInteger(context)));
	case LogicalTypeId::SMALLINT:
		return Value::SMALLINT(NumericCast<int16_t>(json.GetInteger(context)));
	case LogicalTypeId::INTEGER:
		return Value::INTEGER(NumericCast<int32_t>(json.GetInteger(context)));
	case LogicalTypeId::BIGINT:
		return Value::BIGINT(json.GetInteger(context));
	case LogicalTypeId::FLOAT:
		return Value(static_cast<float>(json.GetDouble(context)));
	case LogicalTypeId::DOUBLE:
		return Value::DOUBLE(json.GetDouble(context));
	case LogicalTypeId::VARCHAR:
		return Value(json.GetString(context));
	default:
		throw NotImplementedException("LLM adapter cannot materialize values of type \"%s\" yet", type.ToString());
	}
}

static JsonValue ColumnReferenceToJson(const LogicalGet &get, const BoundColumnRefExpression &column) {
	auto binding_index = column.binding.column_index;
	auto &column_ids = get.GetColumnIds();
	if (binding_index >= column_ids.size()) {
		throw NotImplementedException("LLM adapter cannot push down correlated or generated column reference");
	}
	auto column_index = column_ids[binding_index].GetPrimaryIndex();
	if (column_index >= get.names.size()) {
		throw NotImplementedException("LLM adapter cannot push down virtual column reference");
	}
	return JsonValue::Object({
	    {"kind", JsonValue::String("column")},
	    {"name", JsonValue::String(get.names[column_index])},
	    {"duckdb_type", JsonValue::String(get.returned_types[column_index].ToString())},
	});
}

static JsonValue LiteralToJson(const BoundConstantExpression &constant) {
	return JsonValue::Object({
	    {"kind", JsonValue::String("literal")},
	    {"value", ValueToJson(constant.value)},
	    {"duckdb_type", JsonValue::String(constant.value.type().ToString())},
	});
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

static JsonValue ExpressionToPredicate(const LogicalGet &get, const Expression &expr) {
	switch (expr.GetExpressionClass()) {
	case ExpressionClass::BOUND_COLUMN_REF:
		return ColumnReferenceToJson(get, expr.Cast<BoundColumnRefExpression>());
	case ExpressionClass::BOUND_CONSTANT:
		return LiteralToJson(expr.Cast<BoundConstantExpression>());
	case ExpressionClass::BOUND_CAST: {
		auto &cast = expr.Cast<BoundCastExpression>();
		if (cast.try_cast) {
			throw NotImplementedException("LLM adapter cannot push down TRY_CAST predicates yet");
		}
		return ExpressionToPredicate(get, *cast.child);
	}
	case ExpressionClass::BOUND_COMPARISON: {
		auto &comparison = expr.Cast<BoundComparisonExpression>();
		return JsonValue::Object({
		    {"kind", JsonValue::String("comparison")},
		    {"op", JsonValue::String(ComparisonOperatorToString(expr.GetExpressionType()))},
		    {"left", ExpressionToPredicate(get, *comparison.left)},
		    {"right", ExpressionToPredicate(get, *comparison.right)},
		});
	}
	case ExpressionClass::BOUND_CONJUNCTION: {
		auto &conjunction = expr.Cast<BoundConjunctionExpression>();
		vector<JsonValue> children;
		for (auto &child : conjunction.children) {
			children.push_back(ExpressionToPredicate(get, *child));
		}
		return JsonValue::Object({
		    {"kind", JsonValue::String(expr.GetExpressionType() == ExpressionType::CONJUNCTION_AND ? "and" : "or")},
		    {"children", JsonValue::Array(std::move(children))},
		});
	}
	case ExpressionClass::BOUND_OPERATOR: {
		auto &op = expr.Cast<BoundOperatorExpression>();
		if (expr.GetExpressionType() == ExpressionType::OPERATOR_IS_NULL ||
		    expr.GetExpressionType() == ExpressionType::OPERATOR_IS_NOT_NULL) {
			if (op.children.size() != 1) {
				throw InternalException("Unexpected IS NULL child count");
			}
			return JsonValue::Object({
			    {"kind", JsonValue::String(expr.GetExpressionType() == ExpressionType::OPERATOR_IS_NULL ? "is_null"
			                                                                                            : "is_not_null")},
			    {"expr", ExpressionToPredicate(get, *op.children[0])},
			});
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
	JsonValue Post(ClientContext &, const string &path, const JsonValue &payload) const {
		auto body = SerializeJson(payload);
		return ParseJson(HttpPostJson(parsed_endpoint, path, body));
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
	JsonValue predicate;
	bool has_predicate = false;
	bool has_limit = false;
	idx_t limit = 0;
	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<LlmScanBindData>(catalog, table);
		result->predicate = predicate;
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
	vector<JsonValue> rows;
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
		auto response = client.Post(context, "/v1/catalog/introspect",
		                            JsonValue::Object({
		                                {"type", JsonValue::String("introspect_catalog")},
		                                {"catalog", JsonValue::String(GetName())},
		                                {"checkpoint_ref", JsonValue::String(client.CheckpointRef())},
		                            }));
		return ParseCatalogSnapshot(context, response);
	}
	void ReplaceCatalog(ClientContext &context, const LlmCatalogSnapshot &snapshot) {
		catalog_version = snapshot.version;
		main_schema->ReplaceTables(context, snapshot.tables);
	}
	optional_ptr<CatalogEntry> ApplyCreateTable(CatalogTransaction transaction, BoundCreateTableInfo &info);
	JsonValue Select(ClientContext &context, const JsonValue &query) {
		return client.Post(context, "/v1/query/select",
		                   JsonValue::Object({
		                       {"type", JsonValue::String("select")},
		                       {"catalog_version", JsonValue::String(catalog_version)},
		                       {"query", query},
		                   }));
	}

private:
	void DropSchema(ClientContext &context, DropInfo &info) override {
		throw BinderException("LLM catalog does not support dropping schemas");
	}
	LlmCatalogSnapshot ParseCatalogSnapshot(ClientContext &context, const JsonValue &json) {
		LlmCatalogSnapshot snapshot;
		snapshot.version = json.Require("catalog_version", "catalog snapshot").GetString("catalog_version");
		for (auto &schema_json : json.Require("schemas", "catalog snapshot").GetArray("schemas")) {
			auto schema_name = schema_json.Require("name", "schema").GetString("schema.name");
			if (schema_name != DEFAULT_SCHEMA) {
				throw NotImplementedException("LLM adapter only supports the \"%s\" schema for now",
				                              string(DEFAULT_SCHEMA));
			}
			for (auto &table_json : schema_json.Require("tables", "schema").GetArray("schema.tables")) {
				LlmTableMeta table;
				table.schema = schema_name;
				table.name = table_json.Require("name", "table").GetString("table.name");
				for (auto &column_json : table_json.Require("columns", "table").GetArray("table.columns")) {
					LlmColumnMeta column;
					column.name = column_json.Require("name", "column").GetString("column.name");
					auto type_string = column_json.Require("duckdb_type", "column").GetString("column.duckdb_type");
					column.type = ParseAdapterType(context, type_string);
					column.nullable = column_json.Require("nullable", "column").GetBoolean("column.nullable");
					table.columns.push_back(std::move(column));
				}
				if (auto primary_key = table_json.Get("primary_key")) {
					for (auto &key_json : primary_key->GetArray("table.primary_key")) {
						table.primary_key.push_back(key_json.GetString("primary_key column"));
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

static JsonValue CreateTableOpToJson(const string &catalog_name, BoundCreateTableInfo &info) {
	auto &base = info.Base();
	ValidateCreateTable(base);
	vector<JsonValue> columns;
	idx_t column_idx = 0;
	for (auto &column : base.columns.Logical()) {
		columns.push_back(JsonValue::Object({
		    {"name", JsonValue::String(column.Name())},
		    {"duckdb_type", JsonValue::String(column.Type().ToString())},
		    {"nullable", JsonValue::Boolean(ColumnIsNullable(base, column_idx))},
		    {"default", JsonValue::Null()},
		    {"generated", JsonValue::Boolean(false)},
		}));
		column_idx++;
	}
	vector<JsonValue> primary_key;
	for (auto &column : ExtractPrimaryKey(base)) {
		primary_key.push_back(JsonValue::String(column));
	}
	return JsonValue::Object({
	    {"op", JsonValue::String("create_table")},
	    {"catalog", JsonValue::String(catalog_name)},
	    {"schema", JsonValue::String(base.schema)},
	    {"table", JsonValue::String(base.table)},
	    {"on_conflict", JsonValue::String("error")},
	    {"columns", JsonValue::Array(std::move(columns))},
	    {"primary_key", JsonValue::Array(std::move(primary_key))},
	    {"unique", JsonValue::Array({})},
	    {"checks", JsonValue::Array({})},
	    {"foreign_keys", JsonValue::Array({})},
	});
}

optional_ptr<CatalogEntry> LlmCatalog::ApplyCreateTable(CatalogTransaction transaction, BoundCreateTableInfo &info) {
	if (!transaction.HasContext()) {
		throw InternalException("LLM CREATE TABLE requires a client context");
	}
	auto &context = transaction.GetContext();
	auto op = CreateTableOpToJson(GetName(), info);
	auto response = client.Post(context, "/v1/mutations/apply",
	                            JsonValue::Object({
	                                {"type", JsonValue::String("apply_mutation")},
	                                {"base_catalog_version", JsonValue::String(catalog_version)},
	                                {"operations", JsonValue::Array({std::move(op)})},
	                            }));
	auto status = response.Require("status", "mutation response").GetString("mutation status");
	if (status != "applied") {
		throw IOException("LLM mutation failed with status \"%s\"", status);
	}
	auto snapshot = ParseCatalogSnapshot(context, response.Require("catalog", "mutation response"));
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

static JsonValue BuildProjectionJson(const LlmTableEntry &table, const vector<idx_t> &output_column_ids,
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

	vector<JsonValue> projections;
	for (auto column_id : response_column_ids) {
		auto &column = meta.columns[column_id];
		response_types.push_back(column.type);
		projections.push_back(JsonValue::Object({
		    {"name", JsonValue::String(column.name)},
		    {"duckdb_type", JsonValue::String(column.type.ToString())},
		}));
	}
	for (auto column_id : output_column_ids) {
		auto entry = std::find(response_column_ids.begin(), response_column_ids.end(), column_id);
		if (entry == response_column_ids.end()) {
			throw InternalException("LLM scan output column was not requested from adapter");
		}
		output_to_response.push_back(NumericCast<idx_t>(entry - response_column_ids.begin()));
	}
	return JsonValue::Array(std::move(projections));
}

static unique_ptr<GlobalTableFunctionState> LlmScanInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	auto &bind = input.bind_data->Cast<LlmScanBindData>();
	auto result = make_uniq<LlmScanGlobalState>();
	auto output_column_ids = BuildOutputColumnIds(input.column_indexes, input.projection_ids);
	auto projection = BuildProjectionJson(bind.table, output_column_ids, result->response_types, result->output_types,
	                                      result->output_to_response);
	auto response = bind.catalog.Select(
	    context, JsonValue::Object({
	                 {"schema", JsonValue::String(bind.table.GetMeta().schema)},
	                 {"table", JsonValue::String(bind.table.GetMeta().name)},
	                 {"projection", std::move(projection)},
	                 {"predicate", bind.has_predicate ? bind.predicate : JsonValue::Null()},
	                 {"limit", bind.has_limit ? JsonValue::Number(NumericCast<int64_t>(bind.limit)) : JsonValue::Null()},
	             }));
	auto &columns = response.Require("columns", "select response").GetArray("select columns");
	if (columns.size() != result->response_types.size()) {
		throw IOException("LLM adapter select returned %llu columns, expected %llu", columns.size(),
		                  result->response_types.size());
	}
	for (idx_t i = 0; i < columns.size(); i++) {
		auto returned_type = columns[i].Require("duckdb_type", "select column").GetString("select column type");
		if (returned_type != result->response_types[i].ToString()) {
			throw IOException("LLM adapter select returned type \"%s\" for column %llu, expected \"%s\"", returned_type,
			                  i, result->response_types[i].ToString());
		}
	}
	for (auto &row : response.Require("rows", "select response").GetArray("select rows")) {
		if (row.GetArray("select row").size() != result->response_types.size()) {
			throw IOException("LLM adapter select returned a row with the wrong width");
		}
		result->rows.push_back(row);
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
		auto &row = state.rows[state.offset + row_idx].GetArray("select row");
		for (idx_t col_idx = 0; col_idx < state.output_types.size(); col_idx++) {
			auto response_col_idx = state.output_to_response[col_idx];
			output.SetValue(col_idx, row_idx,
			                JsonToValue(row[response_col_idx], state.output_types[col_idx], "select row value"));
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
	vector<JsonValue> predicates;
	for (auto &filter : filters) {
		predicates.push_back(ExpressionToPredicate(get, *filter));
	}
	bind.has_predicate = true;
	bind.predicate = predicates.size() == 1 ? std::move(predicates[0])
	                                        : JsonValue::Object({
	                                              {"kind", JsonValue::String("and")},
	                                              {"children", JsonValue::Array(std::move(predicates))},
	                                          });
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
	static void Optimize(OptimizerExtensionInput &, unique_ptr<LogicalOperator> &plan) {
		TryPushLimit(plan);
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
