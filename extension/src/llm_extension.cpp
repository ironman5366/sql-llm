#define DUCKDB_EXTENSION_MAIN

#include "llm_extension.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/case_insensitive_map.hpp"
#include "duckdb/common/constants.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/common/index_vector.hpp"
#include "duckdb/common/reference_map.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/main/attached_database.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
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
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/operator/logical_create_table.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/planner/operator/logical_update.hpp"
#include "duckdb/planner/parsed_data/bound_create_table_info.hpp"
#include "duckdb/storage/database_size.hpp"
#include "duckdb/storage/storage_extension.hpp"
#include "duckdb/storage/statistics/base_statistics.hpp"
#include "duckdb/storage/table_storage_info.hpp"
#include "duckdb/transaction/transaction.hpp"
#include "duckdb/transaction/transaction_manager.hpp"

namespace duckdb {

class LlmCatalog;
class LlmSchemaEntry;
class LlmTableEntry;

//===--------------------------------------------------------------------===//
// Transactions
//===--------------------------------------------------------------------===//

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

	ErrorData CommitTransaction(ClientContext &context, Transaction &transaction) override {
		lock_guard<mutex> guard(transaction_lock);
		transactions.erase(transaction);
		return ErrorData();
	}

	void RollbackTransaction(Transaction &transaction) override {
		lock_guard<mutex> guard(transaction_lock);
		transactions.erase(transaction);
	}

	void Checkpoint(ClientContext &context, bool force = false) override {
	}

private:
	mutex transaction_lock;
	reference_map_t<Transaction, unique_ptr<LlmTransaction>> transactions;
};

//===--------------------------------------------------------------------===//
// Table scan
//===--------------------------------------------------------------------===//

struct LlmScanBindData : public TableFunctionData {
	explicit LlmScanBindData(TableCatalogEntry &table_p) : table(table_p) {
	}

	TableCatalogEntry &table;

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<LlmScanBindData>(table);
		result->column_ids = column_ids;
		return std::move(result);
	}

	bool Equals(const FunctionData &other_p) const override {
		auto &other = other_p.Cast<LlmScanBindData>();
		return &table == &other.table;
	}

	bool SupportStatementCache() const override {
		return false;
	}
};

struct LlmScanGlobalState : public GlobalTableFunctionState {
	idx_t MaxThreads() const override {
		return 1;
	}
};

static unique_ptr<GlobalTableFunctionState> LlmScanInitGlobal(ClientContext &context, TableFunctionInitInput &input) {
	return make_uniq<LlmScanGlobalState>();
}

static void LlmScanFunction(ClientContext &context, TableFunctionInput &data, DataChunk &output) {
	throw NotImplementedException("LLM scan not implemented");
}

static double LlmScanProgress(ClientContext &context, const FunctionData *bind_data,
                              const GlobalTableFunctionState *global_state) {
	return 0.0;
}

static unique_ptr<NodeStatistics> LlmScanCardinality(ClientContext &context, const FunctionData *bind_data) {
	return make_uniq<NodeStatistics>(0, 0);
}

static idx_t LlmRowsScanned(GlobalTableFunctionState &global_state, LocalTableFunctionState &local_state) {
	return 0;
}

static bool LlmPushdownExpression(ClientContext &context, const LogicalGet &get, Expression &expr) {
	return true;
}

static BindInfo LlmGetBindInfo(const optional_ptr<FunctionData> bind_data) {
	auto &data = bind_data->Cast<LlmScanBindData>();
	return BindInfo(data.table);
}

static InsertionOrderPreservingMap<string> LlmScanToString(TableFunctionToStringInput &input) {
	auto &data = input.bind_data->Cast<LlmScanBindData>();
	InsertionOrderPreservingMap<string> result;
	result["Table"] = data.table.name;
	result["Backend"] = "llm";
	return result;
}

static TableFunction LlmTableScanFunction() {
	TableFunction function("llm_scan", {}, LlmScanFunction);
	function.init_global = LlmScanInitGlobal;
	function.cardinality = LlmScanCardinality;
	function.rows_scanned = LlmRowsScanned;
	function.pushdown_expression = LlmPushdownExpression;
	function.to_string = LlmScanToString;
	function.table_scan_progress = LlmScanProgress;
	function.get_bind_info = LlmGetBindInfo;
	function.projection_pushdown = true;
	function.filter_pushdown = true;
	function.filter_prune = true;
	return function;
}

//===--------------------------------------------------------------------===//
// Table entry
//===--------------------------------------------------------------------===//

class LlmTableEntry : public TableCatalogEntry {
public:
	LlmTableEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateTableInfo &info)
	    : TableCatalogEntry(catalog, schema, info) {
	}

	unique_ptr<BaseStatistics> GetStatistics(ClientContext &context, column_t column_id) override {
		return nullptr;
	}

	TableFunction GetScanFunction(ClientContext &context, unique_ptr<FunctionData> &bind_data) override {
		bind_data = make_uniq<LlmScanBindData>(*this);
		return LlmTableScanFunction();
	}

	TableStorageInfo GetStorageInfo(ClientContext &context) override {
		TableStorageInfo result;
		result.cardinality = 0;
		return result;
	}

	void BindUpdateConstraints(Binder &binder, LogicalGet &get, LogicalProjection &proj, LogicalUpdate &update,
	                           ClientContext &context) override {
	}
};

//===--------------------------------------------------------------------===//
// Schema entry
//===--------------------------------------------------------------------===//

class LlmSchemaEntry : public SchemaCatalogEntry {
public:
	LlmSchemaEntry(Catalog &catalog, CreateSchemaInfo &info) : SchemaCatalogEntry(catalog, info) {
	}

	optional_ptr<CatalogEntry> CreateTable(CatalogTransaction transaction, BoundCreateTableInfo &info) override {
		auto &base = info.Base();
		auto table_name = base.table;
		auto existing = tables.find(table_name);

		if (existing != tables.end()) {
			switch (base.on_conflict) {
			case OnCreateConflict::IGNORE_ON_CONFLICT:
				return nullptr;
			case OnCreateConflict::REPLACE_ON_CONFLICT:
				tables.erase(existing);
				break;
			case OnCreateConflict::ERROR_ON_CONFLICT:
			case OnCreateConflict::ALTER_ON_CONFLICT:
				throw CatalogException::EntryAlreadyExists(CatalogType::TABLE_ENTRY, table_name);
			}
		}

		auto table = make_uniq<LlmTableEntry>(catalog, *this, base);
		auto result = table.get();
		tables[table_name] = std::move(table);
		return result;
	}

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
		if (type != CatalogType::TABLE_ENTRY) {
			return;
		}
		for (auto &entry : tables) {
			callback(*entry.second);
		}
	}

	void DropEntry(ClientContext &context, DropInfo &info) override {
		if (info.type != CatalogType::TABLE_ENTRY) {
			throw BinderException("LLM catalog only supports dropping tables");
		}
		auto entry = tables.find(info.name);
		if (entry == tables.end()) {
			if (info.if_not_found == OnEntryNotFound::RETURN_NULL) {
				return;
			}
			throw CatalogException("Table with name \"%s\" does not exist", info.name);
		}
		tables.erase(entry);
	}

	optional_ptr<CatalogEntry> LookupEntry(CatalogTransaction transaction, const EntryLookupInfo &lookup_info) override {
		if (lookup_info.GetCatalogType() != CatalogType::TABLE_ENTRY) {
			return nullptr;
		}
		auto entry = tables.find(lookup_info.GetEntryName());
		if (entry == tables.end()) {
			return nullptr;
		}
		return entry->second.get();
	}

private:
	case_insensitive_map_t<unique_ptr<LlmTableEntry>> tables;
};

//===--------------------------------------------------------------------===//
// DML physical operators
//===--------------------------------------------------------------------===//

class LlmWriteOperator : public PhysicalOperator {
public:
	LlmWriteOperator(PhysicalPlan &physical_plan, LogicalOperator &op, string operation_p,
	                 optional_ptr<TableCatalogEntry> table_p)
	    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, op.types, 1),
	      operation(std::move(operation_p)), table(table_p) {
	}

	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override {
		throw NotImplementedException("LLM " + operation + " not implemented");
	}

	SinkResultType Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const override {
		throw NotImplementedException("LLM " + operation + " not implemented");
	}

	SourceResultType GetDataInternal(ExecutionContext &context, DataChunk &chunk,
	                                 OperatorSourceInput &input) const override {
		chunk.SetCardinality(1);
		chunk.SetValue(0, 0, Value::BIGINT(0));
		return SourceResultType::FINISHED;
	}

	bool IsSink() const override {
		return true;
	}

	bool IsSource() const override {
		return true;
	}

	bool ParallelSink() const override {
		return false;
	}

	string GetName() const override {
		return StringUtil::Upper("llm_" + operation);
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override {
		InsertionOrderPreservingMap<string> result;
		if (table) {
			result["Table Name"] = table->name;
		}
		return result;
	}

private:
	string operation;
	optional_ptr<TableCatalogEntry> table;
};

class LlmCreateTableAsOperator : public LlmWriteOperator {
public:
	LlmCreateTableAsOperator(PhysicalPlan &physical_plan, LogicalOperator &op, SchemaCatalogEntry &schema_p,
	                         unique_ptr<BoundCreateTableInfo> info_p)
	    : LlmWriteOperator(physical_plan, op, "create table as", nullptr), schema(schema_p), info(std::move(info_p)) {
	}

	InsertionOrderPreservingMap<string> ParamsToString() const override {
		InsertionOrderPreservingMap<string> result;
		result["Table Name"] = info->Base().table;
		return result;
	}

private:
	SchemaCatalogEntry &schema;
	unique_ptr<BoundCreateTableInfo> info;
};

//===--------------------------------------------------------------------===//
// Catalog
//===--------------------------------------------------------------------===//

class LlmCatalog : public Catalog {
public:
	explicit LlmCatalog(AttachedDatabase &db_p, string path_p) : Catalog(db_p), path(std::move(path_p)) {
	}

	void Initialize(bool load_builtin) override {
		CreateSchemaInfo info;
		info.catalog = GetName();
		info.schema = DEFAULT_SCHEMA;
		main_schema = make_uniq<LlmSchemaEntry>(*this, info);
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
		auto &create = planner.Make<LlmCreateTableAsOperator>(op, op.schema, std::move(op.info));
		create.children.push_back(plan);
		return create;
	}

	PhysicalOperator &PlanInsert(ClientContext &context, PhysicalPlanGenerator &planner, LogicalInsert &op,
	                             optional_ptr<PhysicalOperator> plan) override {
		D_ASSERT(plan);
		auto &insert = planner.Make<LlmWriteOperator>(op, "insert", &op.table);
		insert.children.push_back(*plan);
		return insert;
	}

	PhysicalOperator &PlanDelete(ClientContext &context, PhysicalPlanGenerator &planner, LogicalDelete &op,
	                             PhysicalOperator &plan) override {
		auto &del = planner.Make<LlmWriteOperator>(op, "delete", &op.table);
		del.children.push_back(plan);
		return del;
	}

	PhysicalOperator &PlanUpdate(ClientContext &context, PhysicalPlanGenerator &planner, LogicalUpdate &op,
	                             PhysicalOperator &plan) override {
		auto &update = planner.Make<LlmWriteOperator>(op, "update", &op.table);
		update.children.push_back(plan);
		return update;
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

private:
	void DropSchema(ClientContext &context, DropInfo &info) override {
		throw BinderException("LLM catalog does not support dropping schemas");
	}

private:
	string path;
	unique_ptr<LlmSchemaEntry> main_schema;
};

//===--------------------------------------------------------------------===//
// Storage extension
//===--------------------------------------------------------------------===//

class LlmStorageExtension : public StorageExtension {
public:
	LlmStorageExtension() {
		attach = [](optional_ptr<StorageExtensionInfo>, ClientContext &, AttachedDatabase &db, const string &,
		            AttachInfo &info, AttachOptions &) -> unique_ptr<Catalog> {
			return make_uniq<LlmCatalog>(db, info.path);
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
