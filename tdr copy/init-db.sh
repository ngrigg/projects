#!/bin/bash
# init-db.sh
echo "Connecting to MongoDB and setting up..."
mongosh <<EOF
use tdr_case

// Upsert operation on tasks collection
var upsertResult = db.tasks.updateOne(
  { firstRecord: 1 },
  { \$set: { firstRecord: 1 }},
  { upsert: true }
);
printjson(upsertResult);

// Check if user exists before creating
var userExists = db.getUser("init_user");
if (!userExists) {
  db.createUser({
    user: "init_user",
    pwd: "init_pwd",
    roles: [{role: "readWrite", db: "tdr_case"}]
  });
  print("User created successfully");
} else {
  print("User already exists");
}

EOF
echo "Database and user setup completed."
