#pragma once

#include "duckdb.hpp"
#include "duckdb/catalog/catalog.hpp"
#include "duckdb/catalog/catalog_entry/schema_catalog_entry.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parser/parsed_data/create_schema_info.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/planner/parsed_data/bound_create_table_info.hpp"
#include "duckdb/storage/storage_extension.hpp"
#include "duckdb/transaction/transaction.hpp"
#include "duckdb/transaction/transaction_manager.hpp"

namespace duckdb {

// ---------------------------------------------------------------------------
// Storage info — holds server URL
// ---------------------------------------------------------------------------
struct SqlLlmStorageInfo : public StorageExtensionInfo {
	string server_url;
	explicit SqlLlmStorageInfo(string url) : server_url(std::move(url)) {}
};

// Forward declarations
class SqlLlmSchemaEntry;
class SqlLlmTableEntry;

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------
class SqlLlmCatalog : public Catalog {
public:
	SqlLlmCatalog(AttachedDatabase &db, string server_url);

	string server_url;
	unique_ptr<SqlLlmSchemaEntry> default_schema;

	void Initialize(bool load_builtin) override;
	string GetCatalogType() override { return "sql_llm"; }

	optional_ptr<CatalogEntry> CreateSchema(CatalogTransaction transaction, CreateSchemaInfo &info) override;
	optional_ptr<SchemaCatalogEntry> LookupSchema(CatalogTransaction transaction,
	                                              const EntryLookupInfo &schema_lookup,
	                                              OnEntryNotFound if_not_found) override;
	void ScanSchemas(ClientContext &context, std::function<void(SchemaCatalogEntry &)> callback) override;

	PhysicalOperator &PlanCreateTableAs(ClientContext &context, PhysicalPlanGenerator &planner,
	                                    LogicalCreateTable &op, PhysicalOperator &plan) override;
	PhysicalOperator &PlanInsert(ClientContext &context, PhysicalPlanGenerator &planner, LogicalInsert &op,
	                             optional_ptr<PhysicalOperator> plan) override;
	PhysicalOperator &PlanDelete(ClientContext &context, PhysicalPlanGenerator &planner, LogicalDelete &op,
	                             PhysicalOperator &plan) override;
	PhysicalOperator &PlanUpdate(ClientContext &context, PhysicalPlanGenerator &planner, LogicalUpdate &op,
	                             PhysicalOperator &plan) override;

	DatabaseSize GetDatabaseSize(ClientContext &context) override;
	bool InMemory() override { return true; }
	string GetDBPath() override { return ""; }

	static unique_ptr<Catalog> Attach(optional_ptr<StorageExtensionInfo> storage_info,
	                                  ClientContext &context, AttachedDatabase &db,
	                                  const string &name, AttachInfo &info, AttachOptions &options);

protected:
	void DropSchema(ClientContext &context, DropInfo &info) override;
};

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------
class SqlLlmSchemaEntry : public SchemaCatalogEntry {
public:
	SqlLlmSchemaEntry(Catalog &catalog, CreateSchemaInfo &info, string server_url);

	string server_url;
	// Lifetime management only — NOT a lookup cache
	vector<unique_ptr<SqlLlmTableEntry>> owned_entries;

	void Scan(ClientContext &context, CatalogType type,
	          const std::function<void(CatalogEntry &)> &callback) override;
	void Scan(CatalogType type, const std::function<void(CatalogEntry &)> &callback) override;

	optional_ptr<CatalogEntry> CreateTable(CatalogTransaction transaction, BoundCreateTableInfo &info) override;
	optional_ptr<CatalogEntry> CreateFunction(CatalogTransaction transaction, CreateFunctionInfo &info) override;
	optional_ptr<CatalogEntry> CreateView(CatalogTransaction transaction, CreateViewInfo &info) override;
	optional_ptr<CatalogEntry> CreateSequence(CatalogTransaction transaction, CreateSequenceInfo &info) override;
	optional_ptr<CatalogEntry> CreateTableFunction(CatalogTransaction transaction,
	                                               CreateTableFunctionInfo &info) override;
	optional_ptr<CatalogEntry> CreateCopyFunction(CatalogTransaction transaction,
	                                              CreateCopyFunctionInfo &info) override;
	optional_ptr<CatalogEntry> CreatePragmaFunction(CatalogTransaction transaction,
	                                                CreatePragmaFunctionInfo &info) override;
	optional_ptr<CatalogEntry> CreateCollation(CatalogTransaction transaction, CreateCollationInfo &info) override;
	optional_ptr<CatalogEntry> CreateType(CatalogTransaction transaction, CreateTypeInfo &info) override;
	optional_ptr<CatalogEntry> CreateIndex(CatalogTransaction transaction, CreateIndexInfo &info,
	                                       TableCatalogEntry &table) override;

	optional_ptr<CatalogEntry> LookupEntry(CatalogTransaction transaction,
	                                       const EntryLookupInfo &lookup_info) override;

	void DropEntry(ClientContext &context, DropInfo &info) override;
	void Alter(CatalogTransaction transaction, AlterInfo &info) override;
};

// ---------------------------------------------------------------------------
// Table entry
// ---------------------------------------------------------------------------
class SqlLlmTableEntry : public TableCatalogEntry {
public:
	SqlLlmTableEntry(Catalog &catalog, SchemaCatalogEntry &schema, CreateTableInfo &info, string server_url);

	string server_url;

	TableFunction GetScanFunction(ClientContext &context, unique_ptr<FunctionData> &bind_data) override;
	unique_ptr<BaseStatistics> GetStatistics(ClientContext &context, column_t column_id) override;
	TableStorageInfo GetStorageInfo(ClientContext &context) override;
};

// ---------------------------------------------------------------------------
// Transaction
// ---------------------------------------------------------------------------
class SqlLlmTransaction : public Transaction {
public:
	SqlLlmTransaction(TransactionManager &manager, ClientContext &context, string server_url);
	string server_url;
};

// ---------------------------------------------------------------------------
// Transaction manager
// ---------------------------------------------------------------------------
class SqlLlmTransactionManager : public TransactionManager {
public:
	SqlLlmTransactionManager(AttachedDatabase &db, string server_url);

	string server_url;
	mutex transaction_lock;

	Transaction &StartTransaction(ClientContext &context) override;
	ErrorData CommitTransaction(ClientContext &context, Transaction &transaction) override;
	void RollbackTransaction(Transaction &transaction) override;
	void Checkpoint(ClientContext &context, bool force = false) override;

	static unique_ptr<TransactionManager> Create(optional_ptr<StorageExtensionInfo> storage_info,
	                                             AttachedDatabase &db, Catalog &catalog);

private:
	vector<unique_ptr<SqlLlmTransaction>> transactions;
};

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------
class SqlLlmExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override;
	std::string Name() override;
	std::string Version() const override;
};

} // namespace duckdb
