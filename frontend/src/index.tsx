import React from "react";
import { createRoot } from "react-dom/client";
import QuestionBox from "./question_box";

const App = () => {
  return QuestionBox();
};

const container = document.getElementById("root")!;
const root = createRoot(container);
root.render(<App />);