#define DUCKDB_EXTENSION_MAIN

#include "llm_extension.hpp"

#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/common/constants.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/common/reference_map.hpp"
#include "duckdb/execution/physical_plan_generator.hpp"
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
#include "duckdb/planner/operator/logical_create_table.hpp"
#include "duckdb/planner/operator/logical_delete.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/planner/operator/logical_update.hpp"
#include "duckdb/planner/parsed_data/bound_create_table_info.hpp"
#include "duckdb/storage/database_size.hpp"
#include "duckdb/storage/storage_extension.hpp"
#include "duckdb/transaction/transaction.hpp"
#include "duckdb/transaction/transaction_manager.hpp"

namespace duckdb {

class LlmCatalog;
class LlmSchemaEntry;

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
// Schema entry
//===--------------------------------------------------------------------===//

class LlmSchemaEntry : public SchemaCatalogEntry {
public:
	LlmSchemaEntry(Catalog &catalog, CreateSchemaInfo &info) : SchemaCatalogEntry(catalog, info) {
	}

	optional_ptr<CatalogEntry> CreateTable(CatalogTransaction transaction, BoundCreateTableInfo &info) override {
		throw NotImplementedException("LLM CREATE TABLE requires safetensors-backed catalog metadata");
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
		return;
	}

	void DropEntry(ClientContext &context, DropInfo &info) override {
		throw NotImplementedException("LLM DROP requires safetensors-backed catalog metadata");
	}

	optional_ptr<CatalogEntry> LookupEntry(CatalogTransaction transaction, const EntryLookupInfo &lookup_info) override {
		return nullptr;
	}
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
		throw NotImplementedException("LLM CREATE TABLE AS requires safetensors-backed catalog metadata");
	}

	PhysicalOperator &PlanInsert(ClientContext &context, PhysicalPlanGenerator &planner, LogicalInsert &op,
	                             optional_ptr<PhysicalOperator> plan) override {
		throw NotImplementedException("LLM INSERT requires safetensors-backed catalog metadata");
	}

	PhysicalOperator &PlanDelete(ClientContext &context, PhysicalPlanGenerator &planner, LogicalDelete &op,
	                             PhysicalOperator &plan) override {
		throw NotImplementedException("LLM DELETE requires safetensors-backed catalog metadata");
	}

	PhysicalOperator &PlanUpdate(ClientContext &context, PhysicalPlanGenerator &planner, LogicalUpdate &op,
	                             PhysicalOperator &plan) override {
		throw NotImplementedException("LLM UPDATE requires safetensors-backed catalog metadata");
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
