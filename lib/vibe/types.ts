export interface VibeConfig {
  version?: string;
  architecture?: {
    preferred_pattern?: string;
    forbidden_patterns?: string[];
  };
  code_style?: {
    naming?: Record<string, string>;
    error_handling?: string;
  };
  testing?: {
    framework?: string;
    require_for?: string[];
  };
  qa_rules?: {
    mandatory_evidence?: string[];
    strict_guidelines?: string[];
  };
}
