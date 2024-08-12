import os
import pydoc

def list_modules(package):
    package_dir = package.__path__[0]
    modules = []
    for dirpath, _, filenames in os.walk(package_dir):
        for filename in filenames:
            if filename.endswith('.py') and filename != '__init__.py':
                module = os.path.relpath(os.path.join(dirpath, filename), package_dir)
                module = module[:-3].replace(os.path.sep, '.')
                modules.append(f"{package.__name__}.{module}")
    return modules

if __name__ == "__main__":
    import multigen
    modules = list_modules(multigen)
    modules.insert(0, 'multigen')  # Ensure the package itself is documented first
    
    for module in modules:
        print(f"Generating documentation for {module}...")
        pydoc.writedoc(module)

