from database.connections import get_motherduck_connection

print("Conectando...")

con = get_motherduck_connection("latam_bank")

print("Conectado.")

print("\n=== DATABASES ===")
print(con.sql("SHOW DATABASES").fetchall())

print("\n=== SCHEMAS ===")
print(con.sql("SHOW SCHEMAS").fetchall())

print("\n=== TABLAS DE BRONZE ===")
tablas = con.sql("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'bronze'
    ORDER BY table_name
""").fetchall()

for tabla in tablas:
    print(tabla[0])

print(f"\nNúmero de tablas: {len(tablas)}")



columns = con.sql("""
    SELECT
        table_name,
        column_name,
        data_type,
        ordinal_position
    FROM information_schema.columns
    WHERE table_schema = 'bronze'
    ORDER BY table_name, ordinal_position
""").df()

print(columns)

#exportar columnas y sus relaciones a un archivo CSV

columns.to_csv("columns.csv", index=False)

con.close()


