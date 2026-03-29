import ast

class CodeValidator:
    """Validates the structure and safety of generated Python code."""
    
    ALLOWED_IMPORTS = {'math'}
    REQUIRED_FUNCTION = 'bidding_strategy'

    @staticmethod
    def validate(code: str) -> bool:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        has_strategy_func = False
        
        for node in ast.walk(tree):
            # Check for imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in CodeValidator.ALLOWED_IMPORTS:
                        return False
            elif isinstance(node, ast.ImportFrom):
                if node.module not in CodeValidator.ALLOWED_IMPORTS:
                    return False
            
            # Check for function definition
            if isinstance(node, ast.FunctionDef):
                if node.name == CodeValidator.REQUIRED_FUNCTION:
                    has_strategy_func = True

            # Block dangerous functions (basic check, not exhaustive)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in {'eval', 'exec', 'open', '__import__'}:
                        return False
                elif isinstance(node.func, ast.Attribute):
                    # Prevent os.system etc. though imports should catch this
                    pass

        return has_strategy_func

