import { connect } from "mongoose";

export default async () => {
  try {
    const response = await connect(process.env.MONGO_URI);
    console.log("Connected to", response.connection.name);
  } catch (error) {
    console.error("Error connecting to MongoDB:", error);
  }
};
