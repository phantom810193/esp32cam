BEGIN TRANSACTION;
CREATE TABLE member_profiles (
                    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_label TEXT NOT NULL UNIQUE,
                    member_id TEXT UNIQUE,
                    mall_member_id TEXT,
                    member_status TEXT DEFAULT '有效',
                    joined_at TEXT,
                    points_balance REAL DEFAULT 0,
                    gender TEXT,
                    birth_date TEXT,
                    phone TEXT,
                    email TEXT,
                    address TEXT,
                    occupation TEXT,
                    first_image_filename TEXT,
                    FOREIGN KEY(member_id) REFERENCES members(member_id)
                );
CREATE TABLE members (
                    member_id TEXT PRIMARY KEY,
                    encoding_json TEXT NOT NULL
                );
CREATE TABLE purchases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    member_id TEXT NOT NULL,
                    member_code TEXT NOT NULL,
                    purchased_at TEXT NOT NULL,
                    item TEXT NOT NULL,
                    unit_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    total_price REAL NOT NULL,
                    product_category TEXT NOT NULL,
                    product_code TEXT NOT NULL,
                    product_view_rate REAL NOT NULL,
                    FOREIGN KEY(member_id) REFERENCES members(member_id)
                );
CREATE TABLE upload_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    member_id TEXT NOT NULL,
                    image_filename TEXT,
                    upload_duration REAL NOT NULL,
                    recognition_duration REAL NOT NULL,
                    ad_duration REAL NOT NULL,
                    total_duration REAL NOT NULL,
                    FOREIGN KEY(member_id) REFERENCES members(member_id)
                );
DELETE FROM "sqlite_sequence";
COMMIT;
