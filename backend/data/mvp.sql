BEGIN TRANSACTION;
CREATE TABLE member_profiles (
                    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_label TEXT NOT NULL UNIQUE,
                    name TEXT,
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
INSERT INTO "member_profiles" VALUES(1,'dessert-lover','林悅心','MEME0383FE3AA','ME0001','有效','2021-06-12',1520.0,'女','1988-07-12','0912-345-678','dessertlover@example.com','台北市信義區松壽路10號','甜點教室講師',NULL);
INSERT INTO "member_profiles" VALUES(2,'family-groceries','陳雅雯','MEM692FFD0824','ME0002','有效','2020-09-01',980.0,'女','1990-02-08','0923-556-789','familybuyer@example.com','新北市板橋區文化路100號','幼兒園老師',NULL);
INSERT INTO "member_profiles" VALUES(3,'fitness-enthusiast','張智翔','MEMFITNESS2025','ME0003','有效','2019-11-20',2040.0,'男','1985-04-19','0955-112-233','fitgoer@example.com','台中市西屯區市政北二路88號','企業健身顧問',NULL);
INSERT INTO "member_profiles" VALUES(4,'home-manager','黃珮真','MEMHOMECARE2025','',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
INSERT INTO "member_profiles" VALUES(5,'wellness-gourmet','吳品蓉','MEMHEALTH2025','',NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);
CREATE TABLE members (
                    member_id TEXT PRIMARY KEY,
                    encoding_json TEXT NOT NULL
                );
INSERT INTO "members" VALUES('MEME0383FE3AA','{"vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "signature": "MEME0383FE3AA", "source": "seed"}');
INSERT INTO "members" VALUES('MEM692FFD0824','{"vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "signature": "MEM692FFD0824", "source": "seed"}');
INSERT INTO "members" VALUES('MEMFITNESS2025','{"vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "signature": "MEMFITNESS2025", "source": "seed"}');
INSERT INTO "members" VALUES('MEMHOMECARE2025','{"vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "signature": "MEMHOMECARE2025", "source": "seed"}');
INSERT INTO "members" VALUES('MEMHEALTH2025','{"vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "signature": "MEMHEALTH2025", "source": "seed"}');
CREATE TABLE purchases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    member_id TEXT NOT NULL,
                    member_code TEXT NOT NULL,
                    product_category TEXT NOT NULL DEFAULT '',
                    internal_item_code TEXT NOT NULL DEFAULT '',
                    purchased_at TEXT NOT NULL,
                    item TEXT NOT NULL,
                    unit_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    total_price REAL NOT NULL,
                    FOREIGN KEY(member_id) REFERENCES members(member_id)
                );
INSERT INTO "purchases" VALUES(51,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-001','2025-09-04 10:30','草莓千層蛋糕',320.0,1.0,320.0);
INSERT INTO "purchases" VALUES(52,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-002','2025-09-08 11:41','香草可麗露禮盒',480.0,1.0,480.0);
INSERT INTO "purchases" VALUES(53,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-003','2025-09-12 12:52','抹茶生乳捲',280.0,1.0,280.0);
INSERT INTO "purchases" VALUES(54,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-004','2025-09-16 14:03','精品手沖咖啡豆',520.0,1.0,520.0);
INSERT INTO "purchases" VALUES(55,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-005','2025-09-16 15:14','手作果醬三入組',450.0,1.0,450.0);
INSERT INTO "purchases" VALUES(56,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-006','2025-09-20 11:25','焦糖海鹽布蕾',95.0,2.0,190.0);
INSERT INTO "purchases" VALUES(57,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-007','2025-09-24 11:36','蜂蜜檸檬磅蛋糕',210.0,1.0,210.0);
INSERT INTO "purchases" VALUES(58,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-008','2025-09-28 12:47','法式莓果塔',260.0,1.0,260.0);
INSERT INTO "purchases" VALUES(59,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-009','2025-09-28 13:58','嚴選花草茶禮盒',680.0,1.0,680.0);
INSERT INTO "purchases" VALUES(60,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-010','2025-09-01 15:09','有機燕麥早餐罐',180.0,2.0,360.0);
INSERT INTO "purchases" VALUES(61,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-011','2025-09-05 11:20','生巧克力布朗尼',180.0,2.0,360.0);
INSERT INTO "purchases" VALUES(62,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-012','2025-09-09 11:31','芒果生乳酪杯',150.0,2.0,300.0);
INSERT INTO "purchases" VALUES(63,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-013','2025-09-09 12:42','伯爵茶瑪德蓮',85.0,4.0,340.0);
INSERT INTO "purchases" VALUES(64,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-014','2025-09-13 13:53','冷萃咖啡瓶裝禮盒',360.0,1.0,360.0);
INSERT INTO "purchases" VALUES(65,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-015','2025-09-17 15:04','產地直送有機蔬菜箱',880.0,1.0,880.0);
INSERT INTO "purchases" VALUES(66,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-016','2025-09-21 11:15','紅絲絨杯子蛋糕',120.0,2.0,240.0);
INSERT INTO "purchases" VALUES(67,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-017','2025-09-21 12:26','巴斯克乳酪蛋糕',420.0,1.0,420.0);
INSERT INTO "purchases" VALUES(68,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-018','2025-09-25 12:37','桂花烏龍奶酪',110.0,2.0,220.0);
INSERT INTO "purchases" VALUES(69,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-019','2025-09-01 13:48','季節鮮採綜合水果箱',720.0,1.0,720.0);
INSERT INTO "purchases" VALUES(70,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-020','2025-09-05 14:59','當季鮮採藍莓盒',250.0,1.0,250.0);
INSERT INTO "purchases" VALUES(71,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-021','2025-09-05 11:10','藍莓優格慕斯',135.0,2.0,270.0);
INSERT INTO "purchases" VALUES(72,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-022','2025-09-09 12:21','榛果可麗餅捲',160.0,2.0,320.0);
INSERT INTO "purchases" VALUES(73,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-023','2025-09-13 12:32','柚香乳酪塔',240.0,1.0,240.0);
INSERT INTO "purchases" VALUES(74,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-024','2025-09-17 13:43','溫室小黃瓜三入組',95.0,1.0,95.0);
INSERT INTO "purchases" VALUES(75,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-025','2025-09-17 14:54','冷凍鮭魚切片家庭包',560.0,1.0,560.0);
INSERT INTO "purchases" VALUES(76,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-026','2025-09-21 11:05','抹茶紅豆鬆餅',150.0,2.0,300.0);
INSERT INTO "purchases" VALUES(77,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-027','2025-09-25 12:16','花生黑糖奶酪',105.0,2.0,210.0);
INSERT INTO "purchases" VALUES(78,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-028','2025-09-29 13:27','玫瑰荔枝蛋糕',360.0,1.0,360.0);
INSERT INTO "purchases" VALUES(79,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-029','2025-09-29 13:38','家用環保洗碗精補充包',150.0,2.0,300.0);
INSERT INTO "purchases" VALUES(80,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-030','2025-09-02 14:49','柔感棉質廚房紙巾組',320.0,1.0,320.0);
INSERT INTO "purchases" VALUES(81,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-031','2025-09-06 11:00','太妃焦糖蘋果塔',280.0,1.0,280.0);
INSERT INTO "purchases" VALUES(82,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-032','2025-09-10 12:11','香蕉核桃麵包布丁',180.0,2.0,360.0);
INSERT INTO "purchases" VALUES(83,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-033','2025-09-10 13:22','海鹽奶油司康',90.0,4.0,360.0);
INSERT INTO "purchases" VALUES(84,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-034','2025-09-14 13:33','天然海鹽烹飪罐',150.0,1.0,150.0);
INSERT INTO "purchases" VALUES(85,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-035','2025-09-18 14:44','舒眠香氛蠟燭',320.0,1.0,320.0);
INSERT INTO "purchases" VALUES(86,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-036','2025-09-22 10:55','柑橘乳酪生乳捲',295.0,1.0,295.0);
INSERT INTO "purchases" VALUES(87,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-037','2025-09-22 12:06','法芙娜巧克力塔',340.0,1.0,340.0);
INSERT INTO "purchases" VALUES(88,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-038','2025-09-26 13:17','葡萄柚優格杯',140.0,2.0,280.0);
INSERT INTO "purchases" VALUES(89,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-039','2025-09-30 14:28','有機鮮乳家庭箱',260.0,1.0,260.0);
INSERT INTO "purchases" VALUES(90,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-040','2025-09-04 14:39','無糖優酪乳六入',220.0,1.0,220.0);
INSERT INTO "purchases" VALUES(91,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-041','2025-09-04 10:50','黑糖波士頓派',330.0,1.0,330.0);
INSERT INTO "purchases" VALUES(92,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-042','2025-09-08 12:01','餅乾奶油杯',95.0,3.0,285.0);
INSERT INTO "purchases" VALUES(93,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-043','2025-09-12 13:12','草莓生乳酪塔',260.0,1.0,260.0);
INSERT INTO "purchases" VALUES(94,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-044','2025-09-16 14:23','不沾煎鍋28CM',880.0,1.0,880.0);
INSERT INTO "purchases" VALUES(95,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-045','2025-09-16 14:34','家用濾水壺',520.0,1.0,520.0);
INSERT INTO "purchases" VALUES(96,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-046','2025-09-20 10:45','伯爵奶茶布丁',110.0,2.0,220.0);
INSERT INTO "purchases" VALUES(97,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-047','2025-09-24 11:56','楓糖肉桂捲',85.0,4.0,340.0);
INSERT INTO "purchases" VALUES(98,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-048','2025-09-28 13:07','抹茶巴菲杯',165.0,2.0,330.0);
INSERT INTO "purchases" VALUES(99,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-049','2025-0-28 14:18','多功能香料罐組',360.0,1.0,360.0);
INSERT INTO "purchases" VALUES(100,'MEME0383FE3AA','ME0001','甜點與烘焙','DES-050','2025-09-01 15:29','全麥吐司家庭包',120.0,2.0,240.0);
INSERT INTO "purchases" VALUES(151,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-001','2025-09-05 09:20','幼兒律動課體驗券',680.0,1.0,680.0);
INSERT INTO "purchases" VALUES(152,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-002','2025-09-09 10:31','親子烘焙下午茶套票',1180.0,1.0,1180.0);
INSERT INTO "purchases" VALUES(153,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-003','2025-09-13 11:42','益智積木組',450.0,1.0,450.0);
INSERT INTO "purchases" VALUES(154,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-004','2025-09-17 12:53','家庭健康維他命組',850.0,1.0,850.0);
INSERT INTO "purchases" VALUES(155,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-005','2025-09-17 14:04','週末市集有機蔬菜箱',980.0,1.0,980.0);
INSERT INTO "purchases" VALUES(156,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-006','2025-09-21 10:15','幼幼圖卡教材',320.0,1.0,320.0);
INSERT INTO "purchases" VALUES(157,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-007','2025-09-25 10:26','幼兒園夏令營報名費',5200.0,1.0,5200.0);
INSERT INTO "purchases" VALUES(158,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-008','2025-09-29 11:37','親子瑜伽月票',1680.0,1.0,1680.0);
INSERT INTO "purchases" VALUES(159,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-009','2025-09-29 12:48','家用濾水壺替換濾芯',450.0,2.0,900.0);
INSERT INTO "purchases" VALUES(160,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-010','2025-09-02 13:59','智能體重計',1650.0,1.0,1650.0);
INSERT INTO "purchases" VALUES(161,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-011','2025-09-06 10:10','木製拼圖組',560.0,1.0,560.0);
INSERT INTO "purchases" VALUES(162,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-012','2025-09-10 10:21','幼兒繪本套書',1380.0,1.0,1380.0);
INSERT INTO "purchases" VALUES(163,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-013','2025-09-10 11:32','幼兒科學實驗盒',680.0,1.0,680.0);
INSERT INTO "purchases" VALUES(164,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-014','2025-09-14 12:43','無線吸塵器濾網組',620.0,1.0,620.0);
INSERT INTO "purchases" VALUES(165,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-015','2025-09-18 13:54','家庭常備洗衣精補充包',320.0,3.0,960.0);
INSERT INTO "purchases" VALUES(166,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-016','2025-09-22 10:05','幼兒園延托服務時數',420.0,5.0,2100.0);
INSERT INTO "purchases" VALUES(167,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-017','2025-09-22 11:16','兒童才藝試上課程',780.0,1.0,780.0);
INSERT INTO "purchases" VALUES(168,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-018','2025-09-26 11:27','親子劇場週末票',980.0,1.0,980.0);
INSERT INTO "purchases" VALUES(169,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-019','2025-09-02 12:38','旅行收納壓縮袋組',560.0,1.0,560.0);
INSERT INTO "purchases" VALUES(170,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-020','2025-09-06 13:49','全家早餐穀物禮盒',420.0,2.0,840.0);
INSERT INTO "purchases" VALUES(171,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-021','2025-09-06 10:00','幼兒律動課教材包',460.0,1.0,460.0);
INSERT INTO "purchases" VALUES(172,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-022','2025-09-10 11:11','幼兒園制服組',890.0,1.0,890.0);
INSERT INTO "purchases" VALUES(173,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-023','2025-09-14 11:22','家庭園遊會餐券',350.0,3.0,1050.0);
INSERT INTO "purchases" VALUES(174,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-024','2025-09-18 12:33','季節鮮果禮盒',880.0,1.0,880.0);
INSERT INTO "purchases" VALUES(175,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-025','2025-09-18 13:44','家庭露營炊具套組',1980.0,1.0,1980.0);
INSERT INTO "purchases" VALUES(176,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-026','2025-09-22 09:55','學前美語體驗課',880.0,1.0,880.0);
INSERT INTO "purchases" VALUES(177,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-027','2025-09-26 11:06','幼兒陶土手作課',720.0,1.0,720.0);
INSERT INTO "purchases" VALUES(178,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-028','2025-09-30 12:17','幼兒園校車月票',2800.0,1.0,2800.0);
INSERT INTO "purchases" VALUES(179,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-029','2025-09-30 12:28','家庭號沐浴乳三入組',360.0,1.0,360.0);
INSERT INTO "purchases" VALUES(180,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-030','2025-09-03 13:39','親子保溫水壺雙入',520.0,1.0,520.0);
INSERT INTO "purchases" VALUES(181,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-031','2025-09-07 09:50','幼兒籃球體驗營',1650.0,1.0,1650.0);
INSERT INTO "purchases" VALUES(182,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-032','2025-09-11 11:01','兒童營養午餐組',120.0,10.0,1200.0);
INSERT INTO "purchases" VALUES(183,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-033','2025-09-11 12:12','幼兒園畢業紀念冊預購',550.0,1.0,550.0);
INSERT INTO "purchases" VALUES(184,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-034','2025-09-15 12:23','多用途餐桌防水墊',280.0,1.0,280.0);
INSERT INTO "purchases" VALUES(185,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-035','2025-09-19 13:34','家庭常備繃帶組',180.0,1.0,180.0);
INSERT INTO "purchases" VALUES(186,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-036','2025-09-23 09:45','親子攝影紀念套組',2680.0,1.0,2680.0);
INSERT INTO "purchases" VALUES(187,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-037','2025-09-23 10:56','幼兒戶外探索課程',1350.0,1.0,1350.0);
INSERT INTO "purchases" VALUES(188,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-038','2025-09-27 12:07','兒童舞蹈公演票',880.0,2.0,1760.0);
INSERT INTO "purchases" VALUES(189,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-039','2025-09-01 13:18','親子戶外防曬乳',450.0,1.0,450.0);
INSERT INTO "purchases" VALUES(190,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-040','2025-09-05 13:29','智慧家電延長線',320.0,1.0,320.0);
INSERT INTO "purchases" VALUES(191,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-041','2025-09-05 09:40','幼兒安全防走失背包',960.0,1.0,960.0);
INSERT INTO "purchases" VALUES(192,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-042','2025-09-09 10:51','幼兒園科學週材料包',420.0,1.0,420.0);
INSERT INTO "purchases" VALUES(193,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-043','2025-09-13 12:02','親子閱讀早午餐套票',920.0,1.0,920.0);
INSERT INTO "purchases" VALUES(194,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-044','2025-09-17 13:13','客廳香氛擴香瓶',420.0,1.0,420.0);
INSERT INTO "purchases" VALUES(195,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-045','2025-09-17 13:24','天然洗手乳補充包',260.0,2.0,520.0);
INSERT INTO "purchases" VALUES(196,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-046','2025-09-21 09:35','幼兒園節慶禮盒',680.0,1.0,680.0);
INSERT INTO "purchases" VALUES(197,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-047','2025-09-25 10:46','幼兒園音樂會門票',750.0,2.0,1500.0);
INSERT INTO "purchases" VALUES(198,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-048','2025-09-29 11:57','親子共學木工課',1450.0,1.0,1450.0);
INSERT INTO "purchases" VALUES(199,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-049','2025-09-29 13:08','居家整理收納箱組',580.0,1.0,580.0);
INSERT INTO "purchases" VALUES(200,'MEM692FFD0824','ME0002','親子成長與家庭','FAM-050','2025-09-02 14:19','家庭號即食玉米濃湯',150.0,3.0,450.0);
INSERT INTO "purchases" VALUES(251,'MEMFITNESS2025','ME0003','運動與體能','FIT-001','2025-09-06 07:30','高蛋白乳清粉',1280.0,1.0,1280.0);
INSERT INTO "purchases" VALUES(252,'MEMFITNESS2025','ME0003','運動與體能','FIT-002','2025-09-10 08:41','肌力訓練彈力帶組',420.0,1.0,420.0);
INSERT INTO "purchases" VALUES(253,'MEMFITNESS2025','ME0003','運動與體能','FIT-003','2025-09-14 09:52','健身房月卡',1680.0,1.0,1680.0);
INSERT INTO "purchases" VALUES(254,'MEMFITNESS2025','ME0003','運動與體能','FIT-004','2025-09-18 11:03','運動機能背心',780.0,1.0,780.0);
INSERT INTO "purchases" VALUES(255,'MEMFITNESS2025','ME0003','運動與體能','FIT-005','2025-09-18 12:14','速乾運動毛巾三入組',360.0,1.0,360.0);
INSERT INTO "purchases" VALUES(256,'MEMFITNESS2025','ME0003','運動與體能','FIT-006','2025-09-22 08:25','低脂優格家庭箱',480.0,1.0,480.0);
INSERT INTO "purchases" VALUES(257,'MEMFITNESS2025','ME0003','運動與體能','FIT-007','2025-09-26 08:36','能量燕麥棒禮盒',280.0,1.0,280.0);
INSERT INTO "purchases" VALUES(258,'MEMFITNESS2025','ME0003','運動與體能','FIT-008','2025-09-30 09:47','跑步智能手錶',3680.0,1.0,3680.0);
INSERT INTO "purchases" VALUES(259,'MEMFITNESS2025','ME0003','運動與體能','FIT-009','2025-09-30 10:58','筋膜按摩滾筒',950.0,1.0,950.0);
INSERT INTO "purchases" VALUES(260,'MEMFITNESS2025','ME0003','運動與體能','FIT-010','2025-09-03 12:09','壺鈴訓練組15KG',1680.0,1.0,1680.0);
INSERT INTO "purchases" VALUES(261,'MEMFITNESS2025','ME0003','運動與體能','FIT-011','2025-09-07 08:20','室內跳繩防滑墊',520.0,1.0,520.0);
INSERT INTO "purchases" VALUES(262,'MEMFITNESS2025','ME0003','運動與體能','FIT-012','2025-09-11 08:31','舒肥雞胸冷凍餐',320.0,3.0,960.0);
INSERT INTO "purchases" VALUES(263,'MEMFITNESS2025','ME0003','運動與體能','FIT-013','2025-09-11 09:42','防滑瑜珈墊',980.0,1.0,980.0);
INSERT INTO "purchases" VALUES(264,'MEMFITNESS2025','ME0003','運動與體能','FIT-014','2025-09-15 10:53','運動壓縮襪',420.0,2.0,840.0);
INSERT INTO "purchases" VALUES(265,'MEMFITNESS2025','ME0003','運動與體能','FIT-015','2025-09-19 12:04','健康蔬果汁禮盒',620.0,1.0,620.0);
INSERT INTO "purchases" VALUES(266,'MEMFITNESS2025','ME0003','運動與體能','FIT-016','2025-09-23 08:15','健身料理玻璃保鮮盒組',780.0,1.0,780.0);
INSERT INTO "purchases" VALUES(267,'MEMFITNESS2025','ME0003','運動與體能','FIT-017','2025-09-23 09:26','BCAA胺基酸飲',680.0,1.0,680.0);
INSERT INTO "purchases" VALUES(268,'MEMFITNESS2025','ME0003','運動與體能','FIT-018','2025-09-27 09:37','登山補給凍乾水果',360.0,1.0,360.0);
INSERT INTO "purchases" VALUES(269,'MEMFITNESS2025','ME0003','運動與體能','FIT-019','2025-09-03 10:48','強化護腕',450.0,1.0,450.0);
INSERT INTO "purchases" VALUES(270,'MEMFITNESS2025','ME0003','運動與體能','FIT-020','2025-09-07 11:59','多功能健腹輪',820.0,1.0,820.0);
INSERT INTO "purchases" VALUES(271,'MEMFITNESS2025','ME0003','運動與體能','FIT-021','2025-09-07 08:10','無糖豆漿箱',360.0,1.0,360.0);
INSERT INTO "purchases" VALUES(272,'MEMFITNESS2025','ME0003','運動與體能','FIT-022','2025-09-11 09:21','健康即食藜麥包',420.0,2.0,840.0);
INSERT INTO "purchases" VALUES(273,'MEMFITNESS2025','ME0003','運動與體能','FIT-023','2025-09-15 09:32','室內腳踏車訓練器租賃',1980.0,1.0,1980.0);
INSERT INTO "purchases" VALUES(274,'MEMFITNESS2025','ME0003','運動與體能','FIT-024','2025-09-19 10:43','全穀饅頭六入',150.0,2.0,300.0);
INSERT INTO "purchases" VALUES(275,'MEMFITNESS2025','ME0003','運動與體能','FIT-025','2025-09-19 11:54','運動水壺雙入組',520.0,1.0,520.0);
INSERT INTO "purchases" VALUES(276,'MEMFITNESS2025','ME0003','運動與體能','FIT-026','2025-09-23 08:05','極速冷感運動帽',680.0,1.0,680.0);
INSERT INTO "purchases" VALUES(277,'MEMFITNESS2025','ME0003','運動與體能','FIT-027','2025-09-27 09:16','防滑健身手套',420.0,1.0,420.0);
INSERT INTO "purchases" VALUES(278,'MEMFITNESS2025','ME0003','運動與體能','FIT-028','2025-09-31 10:27','戶外越野襪三入',380.0,1.0,380.0);
INSERT INTO "purchases" VALUES(279,'MEMFITNESS2025','ME0003','運動與體能','FIT-029','2025-09-31 10:38','植物蛋白飲禮盒',560.0,1.0,560.0);
INSERT INTO "purchases" VALUES(280,'MEMFITNESS2025','ME0003','運動與體能','FIT-030','2025-09-04 11:49','冷壓橄欖油家庭瓶',680.0,1.0,680.0);
INSERT INTO "purchases" VALUES(281,'MEMFITNESS2025','ME0003','運動與體能','FIT-031','2025-09-08 08:00','有機藍莓盒',220.0,1.0,220.0);
INSERT INTO "purchases" VALUES(282,'MEMFITNESS2025','ME0003','運動與體能','FIT-032','2025-09-12 09:11','智能體脂計',1480.0,1.0,1480.0);
INSERT INTO "purchases" VALUES(283,'MEMFITNESS2025','ME0003','運動與體能','FIT-033','2025-09-12 10:22','高纖蔬菜湯組',360.0,2.0,720.0);
INSERT INTO "purchases" VALUES(284,'MEMFITNESS2025','ME0003','運動與體能','FIT-034','2025-09-16 10:33','雞蛋白營養飲',420.0,1.0,420.0);
INSERT INTO "purchases" VALUES(285,'MEMFITNESS2025','ME0003','運動與體能','FIT-035','2025-09-20 11:44','運動耳機防汗版',2980.0,1.0,2980.0);
INSERT INTO "purchases" VALUES(286,'MEMFITNESS2025','ME0003','運動與體能','FIT-036','2025-09-24 07:55','夜跑LED臂帶',320.0,1.0,320.0);
INSERT INTO "purchases" VALUES(287,'MEMFITNESS2025','ME0003','運動與體能','FIT-037','2025-09-24 09:06','山藥雞湯即食包',260.0,2.0,520.0);
INSERT INTO "purchases" VALUES(288,'MEMFITNESS2025','ME0003','運動與體能','FIT-038','2025-09-28 10:17','家用洗碗機洗劑',450.0,1.0,450.0);
INSERT INTO "purchases" VALUES(289,'MEMFITNESS2025','ME0003','運動與體能','FIT-039','2025-09-02 11:28','天然洗衣精補充包',320.0,2.0,640.0);
INSERT INTO "purchases" VALUES(290,'MEMFITNESS2025','ME0003','運動與體能','FIT-040','2025-09-06 11:39','旅行收納健身包',880.0,1.0,880.0);
INSERT INTO "purchases" VALUES(291,'MEMFITNESS2025','ME0003','運動與體能','FIT-041','2025-09-06 07:50','居家伸展彈力椅',1350.0,1.0,1350.0);
INSERT INTO "purchases" VALUES(292,'MEMFITNESS2025','ME0003','運動與體能','FIT-042','2025-09-10 09:01','碳酸鎂粉補充包',220.0,1.0,220.0);
INSERT INTO "purchases" VALUES(293,'MEMFITNESS2025','ME0003','運動與體能','FIT-043','2025-09-14 10:12','海鹽堅果能量包',260.0,2.0,520.0);
INSERT INTO "purchases" VALUES(294,'MEMFITNESS2025','ME0003','運動與體能','FIT-044','2025-09-18 11:23','綜合沙拉葉家庭包',180.0,2.0,360.0);
INSERT INTO "purchases" VALUES(295,'MEMFITNESS2025','ME0003','運動與體能','FIT-045','2025-09-18 11:34','舒眠草本茶禮盒',520.0,1.0,520.0);
INSERT INTO "purchases" VALUES(296,'MEMFITNESS2025','ME0003','運動與體能','FIT-046','2025-09-22 07:45','多功能果汁機',2280.0,1.0,2280.0);
INSERT INTO "purchases" VALUES(297,'MEMFITNESS2025','ME0003','運動與體能','FIT-047','2025-09-26 08:56','冬季保暖運動外套',1980.0,1.0,1980.0);
INSERT INTO "purchases" VALUES(298,'MEMFITNESS2025','ME0003','運動與體能','FIT-048','2025-09-30 10:07','戶外蛋白餅乾',180.0,2.0,360.0);
INSERT INTO "purchases" VALUES(299,'MEMFITNESS2025','ME0003','運動與體能','FIT-049','2025-09-30 11:18','家用保溫餐盒',560.0,1.0,560.0);
INSERT INTO "purchases" VALUES(300,'MEMFITNESS2025','ME0003','運動與體能','FIT-050','2025-09-03 12:29','伸展瑜珈磚',360.0,1.0,360.0);
INSERT INTO "purchases" VALUES(353,'MEMHOMECARE2025','','居家生活','HOM-001','2025-09-08 08:45','多功能電鍋',1680.0,1.0,1680.0);
INSERT INTO "purchases" VALUES(354,'MEMHOMECARE2025','','居家生活','HOM-002','2025-09-12 09:56','家庭保鮮盒12件組',520.0,1.0,520.0);
INSERT INTO "purchases" VALUES(355,'MEMHOMECARE2025','','居家生活','HOM-003','2025-09-16 11:07','天然洗衣精補充包',320.0,3.0,960.0);
INSERT INTO "purchases" VALUES(356,'MEMHOMECARE2025','','居家生活','HOM-004','2025-09-20 12:18','兒童學習餐具組',420.0,1.0,420.0);
INSERT INTO "purchases" VALUES(357,'MEMHOMECARE2025','','居家生活','HOM-005','2025-09-20 13:29','客廳防滑地墊',680.0,1.0,680.0);
INSERT INTO "purchases" VALUES(358,'MEMHOMECARE2025','','居家生活','HOM-006','2025-09-24 09:40','智能掃地機濾網',450.0,1.0,450.0);
INSERT INTO "purchases" VALUES(359,'MEMHOMECARE2025','','居家生活','HOM-007','2025-09-28 09:51','廚房去油劑雙入',260.0,1.0,260.0);
INSERT INTO "purchases" VALUES(360,'MEMHOMECARE2025','','居家生活','HOM-008','2025-09-01 11:02','家庭號衛生紙箱',320.0,1.0,320.0);
INSERT INTO "purchases" VALUES(361,'MEMHOMECARE2025','','居家生活','HOM-009','2025-09-01 12:13','有機雞蛋家庭盒',180.0,2.0,360.0);
INSERT INTO "purchases" VALUES(362,'MEMHOMECARE2025','','居家生活','HOM-010','2025-09-05 13:24','當季蔬菜禮籃',520.0,1.0,520.0);
INSERT INTO "purchases" VALUES(363,'MEMHOMECARE2025','','居家生活','HOM-011','2025-09-09 09:35','兒童英文繪本組',680.0,1.0,680.0);
INSERT INTO "purchases" VALUES(364,'MEMHOMECARE2025','','居家生活','HOM-012','2025-09-13 09:46','烘焙常備粉組',260.0,2.0,520.0);
INSERT INTO "purchases" VALUES(365,'MEMHOMECARE2025','','居家生活','HOM-013','2025-09-13 10:57','廚房紙巾超值組',360.0,1.0,360.0);
INSERT INTO "purchases" VALUES(366,'MEMHOMECARE2025','','居家生活','HOM-014','2025-09-17 12:08','家庭常備藥品組',480.0,1.0,480.0);
INSERT INTO "purchases" VALUES(367,'MEMHOMECARE2025','','居家生活','HOM-015','2025-09-21 13:19','保溫便當盒雙層',420.0,1.0,420.0);
INSERT INTO "purchases" VALUES(368,'MEMHOMECARE2025','','居家生活','HOM-016','2025-09-25 09:30','兒童益智拼圖',380.0,1.0,380.0);
INSERT INTO "purchases" VALUES(369,'MEMHOMECARE2025','','居家生活','HOM-017','2025-09-25 10:41','天然酵素清潔液',450.0,1.0,450.0);
INSERT INTO "purchases" VALUES(370,'MEMHOMECARE2025','','居家生活','HOM-018','2025-09-01 10:52','家庭車用收納箱',520.0,1.0,520.0);
INSERT INTO "purchases" VALUES(371,'MEMHOMECARE2025','','居家生活','HOM-019','2025-09-05 12:03','香氛洗手乳三入',320.0,1.0,320.0);
INSERT INTO "purchases" VALUES(372,'MEMHOMECARE2025','','居家生活','HOM-020','2025-09-09 13:14','客廳抱枕套組',420.0,1.0,420.0);
INSERT INTO "purchases" VALUES(373,'MEMHOMECARE2025','','居家生活','HOM-021','2025-09-09 09:25','烘碗機除菌濾網',280.0,1.0,280.0);
INSERT INTO "purchases" VALUES(374,'MEMHOMECARE2025','','居家生活','HOM-022','2025-09-13 10:36','兒童室內拖鞋',260.0,2.0,520.0);
INSERT INTO "purchases" VALUES(375,'MEMHOMECARE2025','','居家生活','HOM-023','2025-09-17 10:47','家庭急救包',620.0,1.0,620.0);
INSERT INTO "purchases" VALUES(376,'MEMHOMECARE2025','','居家生活','HOM-024','2025-09-21 11:58','氣炸鍋烘烤紙',150.0,2.0,300.0);
INSERT INTO "purchases" VALUES(377,'MEMHOMECARE2025','','居家生活','HOM-025','2025-09-21 13:09','親子烹飪課程券',1280.0,1.0,1280.0);
INSERT INTO "purchases" VALUES(378,'MEMHOMECARE2025','','居家生活','HOM-026','2025-09-25 09:20','冷凍餃子家庭包',320.0,2.0,640.0);
INSERT INTO "purchases" VALUES(379,'MEMHOMECARE2025','','居家生活','HOM-027','2025-09-29 10:31','早餐穀片超值箱',360.0,1.0,360.0);
INSERT INTO "purchases" VALUES(380,'MEMHOMECARE2025','','居家生活','HOM-028','2025-09-02 11:42','保鮮袋100入',180.0,1.0,180.0);
INSERT INTO "purchases" VALUES(381,'MEMHOMECARE2025','','居家生活','HOM-029','2025-09-02 11:53','兒童成長牛奶',420.0,2.0,840.0);
INSERT INTO "purchases" VALUES(382,'MEMHOMECARE2025','','居家生活','HOM-030','2025-09-06 13:04','家用滅菌噴霧',380.0,1.0,380.0);
INSERT INTO "purchases" VALUES(383,'MEMHOMECARE2025','','居家生活','HOM-031','2025-09-10 09:15','天然蜂蜜禮盒',560.0,1.0,560.0);
INSERT INTO "purchases" VALUES(384,'MEMHOMECARE2025','','居家生活','HOM-032','2025-09-14 10:26','客廳收納籃組',450.0,1.0,450.0);
INSERT INTO "purchases" VALUES(385,'MEMHOMECARE2025','','居家生活','HOM-033','2025-09-14 11:37','家庭號冷凍鮭魚',520.0,1.0,520.0);
INSERT INTO "purchases" VALUES(386,'MEMHOMECARE2025','','居家生活','HOM-034','2025-09-18 11:48','季節水果拼盤',420.0,1.0,420.0);
INSERT INTO "purchases" VALUES(387,'MEMHOMECARE2025','','居家生活','HOM-035','2025-09-22 12:59','小家庭燒烤盤',780.0,1.0,780.0);
INSERT INTO "purchases" VALUES(388,'MEMHOMECARE2025','','居家生活','HOM-036','2025-09-26 09:10','兒童畫畫教材組',360.0,1.0,360.0);
INSERT INTO "purchases" VALUES(389,'MEMHOMECARE2025','','居家生活','HOM-037','2025-09-26 10:21','除濕包超值組',280.0,2.0,560.0);
INSERT INTO "purchases" VALUES(390,'MEMHOMECARE2025','','居家生活','HOM-038','2025-09-30 11:32','家庭用吸塵器濾網',420.0,1.0,420.0);
INSERT INTO "purchases" VALUES(391,'MEMHOMECARE2025','','居家生活','HOM-039','2025-09-04 12:43','親子桌遊禮盒',680.0,1.0,680.0);
INSERT INTO "purchases" VALUES(392,'MEMHOMECARE2025','','居家生活','HOM-040','2025-09-08 12:54','天然醬油組',320.0,1.0,320.0);
INSERT INTO "purchases" VALUES(393,'MEMHOMECARE2025','','居家生活','HOM-041','2025-09-08 09:05','家庭號優格桶',280.0,1.0,280.0);
INSERT INTO "purchases" VALUES(394,'MEMHOMECARE2025','','居家生活','HOM-042','2025-09-12 10:16','舒眠草本茶',260.0,2.0,520.0);
INSERT INTO "purchases" VALUES(395,'MEMHOMECARE2025','','居家生活','HOM-043','2025-09-16 11:27','有機米禮盒',680.0,1.0,680.0);
INSERT INTO "purchases" VALUES(396,'MEMHOMECARE2025','','居家生活','HOM-044','2025-09-20 12:38','保暖親子毛毯',780.0,1.0,780.0);
INSERT INTO "purchases" VALUES(397,'MEMHOMECARE2025','','居家生活','HOM-045','2025-09-20 12:49','兒童雨衣靴組',620.0,1.0,620.0);
INSERT INTO "purchases" VALUES(398,'MEMHOMECARE2025','','居家生活','HOM-046','2025-09-24 09:00','家庭常備電池組',360.0,1.0,360.0);
INSERT INTO "purchases" VALUES(399,'MEMHOMECARE2025','','居家生活','HOM-047','2025-09-28 10:11','廚房玻璃調味罐',280.0,1.0,280.0);
INSERT INTO "purchases" VALUES(400,'MEMHOMECARE2025','','居家生活','HOM-048','2025-09-01 11:22','居家香氛噴霧',450.0,1.0,450.0);
INSERT INTO "purchases" VALUES(401,'MEMHOMECARE2025','','居家生活','HOM-049','2025-09-01 12:33','蔬果保鮮盒組',360.0,1.0,360.0);
INSERT INTO "purchases" VALUES(402,'MEMHOMECARE2025','','居家生活','HOM-050','2025-09-05 13:44','家庭烘焙模具',320.0,1.0,320.0);
INSERT INTO "purchases" VALUES(403,'MEMHOMECARE2025','','居家生活','HOM-051','2025-09-09 08:55','親子野餐籃',520.0,1.0,520.0);
INSERT INTO "purchases" VALUES(404,'MEMHOMECARE2025','','居家生活','HOM-052','2025-09-13 10:06','天然洗碗皂',220.0,2.0,440.0);
INSERT INTO "purchases" VALUES(457,'MEMHEALTH2025','','健康食尚','HLT-001','2025-09-09 09:10','有機冷壓亞麻仁油',620.0,1.0,620.0);
INSERT INTO "purchases" VALUES(458,'MEMHEALTH2025','','健康食尚','HLT-002','2025-09-13 10:21','高纖燕麥片禮盒',360.0,1.0,360.0);
INSERT INTO "purchases" VALUES(459,'MEMHEALTH2025','','健康食尚','HLT-003','2025-09-17 11:32','綜合堅果禮罐',520.0,1.0,520.0);
INSERT INTO "purchases" VALUES(460,'MEMHEALTH2025','','健康食尚','HLT-004','2025-09-21 12:43','植物基蛋白飲',680.0,1.0,680.0);
INSERT INTO "purchases" VALUES(461,'MEMHEALTH2025','','健康食尚','HLT-005','2025-09-21 13:54','綠拿鐵冷壓汁',260.0,2.0,520.0);
INSERT INTO "purchases" VALUES(462,'MEMHEALTH2025','','健康食尚','HLT-006','2025-09-25 10:05','益生菌粉末盒',820.0,1.0,820.0);
INSERT INTO "purchases" VALUES(463,'MEMHEALTH2025','','健康食尚','HLT-007','2025-09-29 10:16','有機羽衣甘藍',220.0,2.0,440.0);
INSERT INTO "purchases" VALUES(464,'MEMHEALTH2025','','健康食尚','HLT-008','2025-09-02 11:27','低溫烘焙杏仁',360.0,1.0,360.0);
INSERT INTO "purchases" VALUES(465,'MEMHEALTH2025','','健康食尚','HLT-009','2025-09-02 12:38','糙米能量棒',280.0,2.0,560.0);
INSERT INTO "purchases" VALUES(466,'MEMHEALTH2025','','健康食尚','HLT-010','2025-09-06 13:49','高鈣無糖豆漿',320.0,1.0,320.0);
INSERT INTO "purchases" VALUES(467,'MEMHEALTH2025','','健康食尚','HLT-011','2025-09-10 10:00','藜麥綜合穀物飯',380.0,2.0,760.0);
INSERT INTO "purchases" VALUES(468,'MEMHEALTH2025','','健康食尚','HLT-012','2025-09-14 10:11','低GI紫米麵包',260.0,2.0,520.0);
INSERT INTO "purchases" VALUES(469,'MEMHEALTH2025','','健康食尚','HLT-013','2025-09-14 11:22','海藻鈣膠囊',780.0,1.0,780.0);
INSERT INTO "purchases" VALUES(470,'MEMHEALTH2025','','健康食尚','HLT-014','2025-09-18 12:33','有機甜菜根粉',420.0,1.0,420.0);
INSERT INTO "purchases" VALUES(471,'MEMHEALTH2025','','健康食尚','HLT-015','2025-09-22 13:44','天然莓果乾',320.0,2.0,640.0);
INSERT INTO "purchases" VALUES(472,'MEMHEALTH2025','','健康食尚','HLT-016','2025-09-26 09:55','膳食纖維飲品',450.0,1.0,450.0);
INSERT INTO "purchases" VALUES(473,'MEMHEALTH2025','','健康食尚','HLT-017','2025-09-26 11:06','有機小農蔬菜箱',880.0,1.0,880.0);
INSERT INTO "purchases" VALUES(474,'MEMHEALTH2025','','健康食尚','HLT-018','2025-09-02 11:17','優格發酵菌粉',350.0,1.0,350.0);
INSERT INTO "purchases" VALUES(475,'MEMHEALTH2025','','健康食尚','HLT-019','2025-09-06 12:28','天然蜂膠滴劑',560.0,1.0,560.0);
INSERT INTO "purchases" VALUES(476,'MEMHEALTH2025','','健康食尚','HLT-020','2025-09-10 13:39','低溫烘焙腰果',420.0,1.0,420.0);
INSERT INTO "purchases" VALUES(477,'MEMHEALTH2025','','健康食尚','HLT-021','2025-09-10 09:50','紅藜健康米',420.0,1.0,420.0);
INSERT INTO "purchases" VALUES(478,'MEMHEALTH2025','','健康食尚','HLT-022','2025-09-14 11:01','綠茶多酚飲',320.0,2.0,640.0);
INSERT INTO "purchases" VALUES(479,'MEMHEALTH2025','','健康食尚','HLT-023','2025-09-18 11:12','有機高麗菜',160.0,2.0,320.0);
INSERT INTO "purchases" VALUES(480,'MEMHEALTH2025','','健康食尚','HLT-024','2025-09-22 12:23','全穀燕麥奶',280.0,2.0,560.0);
INSERT INTO "purchases" VALUES(481,'MEMHEALTH2025','','健康食尚','HLT-025','2025-09-22 13:34','天然蔓越莓汁',360.0,2.0,720.0);
INSERT INTO "purchases" VALUES(482,'MEMHEALTH2025','','健康食尚','HLT-026','2025-09-26 09:45','有機黑芝麻粉',380.0,1.0,380.0);
INSERT INTO "purchases" VALUES(483,'MEMHEALTH2025','','健康食尚','HLT-027','2025-09-30 10:56','暖薑黑糖飲',280.0,2.0,560.0);
INSERT INTO "purchases" VALUES(484,'MEMHEALTH2025','','健康食尚','HLT-028','2025-09-03 12:07','高蛋白豆腐組',220.0,2.0,440.0);
INSERT INTO "purchases" VALUES(485,'MEMHEALTH2025','','健康食尚','HLT-029','2025-09-03 12:18','冷壓胡蘿蔔汁',260.0,2.0,520.0);
INSERT INTO "purchases" VALUES(486,'MEMHEALTH2025','','健康食尚','HLT-030','2025-09-07 13:29','綠色蔬果粉',520.0,1.0,520.0);
INSERT INTO "purchases" VALUES(487,'MEMHEALTH2025','','健康食尚','HLT-031','2025-09-11 09:40','膠原蛋白飲',780.0,1.0,780.0);
INSERT INTO "purchases" VALUES(488,'MEMHEALTH2025','','健康食尚','HLT-032','2025-09-15 10:51','天然薄荷茶',260.0,2.0,520.0);
INSERT INTO "purchases" VALUES(489,'MEMHEALTH2025','','健康食尚','HLT-033','2025-09-15 12:02','發芽糙米',320.0,2.0,640.0);
INSERT INTO "purchases" VALUES(490,'MEMHEALTH2025','','健康食尚','HLT-034','2025-09-19 12:13','有機酪梨禮盒',620.0,1.0,620.0);
INSERT INTO "purchases" VALUES(491,'MEMHEALTH2025','','健康食尚','HLT-035','2025-09-23 13:24','全植營養補充錠',980.0,1.0,980.0);
INSERT INTO "purchases" VALUES(492,'MEMHEALTH2025','','健康食尚','HLT-036','2025-09-27 09:35','健康烤地瓜片',180.0,2.0,360.0);
INSERT INTO "purchases" VALUES(493,'MEMHEALTH2025','','健康食尚','HLT-037','2025-09-27 10:46','有機藍莓醬',320.0,1.0,320.0);
INSERT INTO "purchases" VALUES(494,'MEMHEALTH2025','','健康食尚','HLT-038','2025-09-01 11:57','燕麥豆奶布丁',260.0,2.0,520.0);
INSERT INTO "purchases" VALUES(495,'MEMHEALTH2025','','健康食尚','HLT-039','2025-09-05 13:08','純淨礦泉水箱',280.0,1.0,280.0);
INSERT INTO "purchases" VALUES(496,'MEMHEALTH2025','','健康食尚','HLT-040','2025-09-09 13:19','天然葡萄籽油',560.0,1.0,560.0);
INSERT INTO "purchases" VALUES(497,'MEMHEALTH2025','','健康食尚','HLT-041','2025-09-09 09:30','無糖椰子水',320.0,2.0,640.0);
INSERT INTO "purchases" VALUES(498,'MEMHEALTH2025','','健康食尚','HLT-042','2025-09-13 10:41','高纖蒟蒻麵',280.0,2.0,560.0);
INSERT INTO "purchases" VALUES(499,'MEMHEALTH2025','','健康食尚','HLT-043','2025-09-17 11:52','有機南瓜',180.0,2.0,360.0);
INSERT INTO "purchases" VALUES(500,'MEMHEALTH2025','','健康食尚','HLT-044','2025-09-21 13:03','保溫隨行杯',420.0,1.0,420.0);
INSERT INTO "purchases" VALUES(501,'MEMHEALTH2025','','健康食尚','HLT-045','2025-09-21 13:14','天然洗衣粉',320.0,1.0,320.0);
INSERT INTO "purchases" VALUES(502,'MEMHEALTH2025','','健康食尚','HLT-046','2025-09-25 09:25','香草舒眠枕噴霧',450.0,1.0,450.0);
INSERT INTO "purchases" VALUES(503,'MEMHEALTH2025','','健康食尚','HLT-047','2025-09-29 10:36','健康烹飪蒸籠',680.0,1.0,680.0);
INSERT INTO "purchases" VALUES(504,'MEMHEALTH2025','','健康食尚','HLT-048','2025-09-02 11:47','有機鷹嘴豆',260.0,2.0,520.0);
INSERT INTO "purchases" VALUES(505,'MEMHEALTH2025','','健康食尚','HLT-049','2025-09-02 12:58','純素黑巧克力',320.0,1.0,320.0);
INSERT INTO "purchases" VALUES(506,'MEMHEALTH2025','','健康食尚','HLT-050','2025-09-06 14:09','天然洗碗精',280.0,1.0,280.0);
INSERT INTO "purchases" VALUES(507,'MEMHEALTH2025','','健康食尚','HLT-051','2025-09-10 09:20','冷壓椰子油',520.0,1.0,520.0);
INSERT INTO "purchases" VALUES(508,'MEMHEALTH2025','','健康食尚','HLT-052','2025-09-14 10:31','有機檸檬禮盒',360.0,1.0,360.0);
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
INSERT INTO "sqlite_sequence" VALUES('member_profiles',5);
INSERT INTO "sqlite_sequence" VALUES('purchases',508);
COMMIT;
