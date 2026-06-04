"""
Tests for all tools.
"""

import pytest
from src.tools.base import BaseTool, ToolResult, ToolError, ToolRegistry
from src.tools.calculator_tool import CalculatorTool
from src.tools.shell_tool import ShellTool
from src.tools.search_tool import SearchTool
from src.tools.file_tool import FileTool
from src.tools.python_tool import PythonTool
from src.tools.browser_tool import BrowserTool
from src.tools.api_tool import APITool


class TestToolResult:
    """Tests for ToolResult."""

    def test_creation_success(self):
        result = ToolResult(success=True, data={"key": "value"})
        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None

    def test_creation_failure(self):
        result = ToolResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"

    def test_to_dict(self):
        result = ToolResult(success=True, data="test", tool_name="my_tool")
        d = result.to_dict()
        assert d["success"] is True
        assert d["data"] == "test"
        assert d["tool_name"] == "my_tool"
        assert "timestamp" in d


class TestToolError:
    """Tests for ToolError."""

    def test_creation(self):
        err = ToolError("test error")
        assert str(err) == "test error"

    def test_is_exception(self):
        assert issubclass(ToolError, Exception)


class TestBaseTool:
    """Tests for BaseTool abstract class."""

    def test_validate_params_success(self):
        class MockTool(BaseTool):
            name = "mock"
            required_params = ["a", "b"]
            async def execute(self, **kwargs):
                return ToolResult(success=True)

        tool = MockTool()
        tool.validate_params(a=1, b=2)  # Should not raise

    def test_validate_params_missing(self):
        class MockTool(BaseTool):
            name = "mock"
            required_params = ["a", "b"]
            async def execute(self, **kwargs):
                return ToolResult(success=True)

        tool = MockTool()
        with pytest.raises(ToolError, match="Missing required parameters"):
            tool.validate_params(a=1)

    def test_enabled_default(self):
        class MockTool(BaseTool):
            name = "mock"
            async def execute(self, **kwargs):
                return ToolResult(success=True)

        tool = MockTool()
        assert tool.enabled is True

    def test_disable_tool(self):
        class MockTool(BaseTool):
            name = "mock"
            async def execute(self, **kwargs):
                return ToolResult(success=True)

        tool = MockTool()
        tool.enabled = False
        assert tool.enabled is False

    def test_get_definition(self):
        class MockTool(BaseTool):
            name = "mock"
            description = "A mock tool"
            category = "test"
            required_params = ["x"]
            optional_params = ["y"]
            async def execute(self, **kwargs):
                return ToolResult(success=True)

        tool = MockTool()
        d = tool.get_definition()
        assert d["name"] == "mock"
        assert d["description"] == "A mock tool"
        assert d["required_params"] == ["x"]


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_and_get(self):
        registry = ToolRegistry()
        calc = CalculatorTool()
        registry.register(calc)
        assert registry.get("calculator") is calc

    def test_register_multiple(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(ShellTool())
        assert len(registry) == 2

    def test_unregister(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.unregister("calculator")
        assert registry.get("calculator") is None

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        registry.register(ShellTool())
        tools = registry.list_tools()
        assert "calculator" in tools
        assert "shell" in tools

    def test_contains(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        assert "calculator" in registry
        assert "shell" not in registry

    def test_get_definitions(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        defs = registry.get_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "calculator"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        result = await registry.execute("calculator", expression="2 + 3")
        assert result.success is True
        assert result.data["result"] == 5

    @pytest.mark.asyncio
    async def test_execute_not_found(self):
        registry = ToolRegistry()
        result = await registry.execute("nonexistent")
        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_execute_disabled(self):
        registry = ToolRegistry()
        calc = CalculatorTool()
        calc.enabled = False
        registry.register(calc)
        result = await registry.execute("calculator", expression="2 + 3")
        assert result.success is False
        assert "disabled" in result.error


class TestCalculatorTool:
    """Tests for CalculatorTool."""

    @pytest.mark.asyncio
    async def test_basic_addition(self):
        tool = CalculatorTool()
        result = await tool.safe_execute(expression="2 + 3")
        assert result.success is True
        assert result.data["result"] == 5

    @pytest.mark.asyncio
    async def test_basic_subtraction(self):
        tool = CalculatorTool()
        result = await tool.safe_execute(expression="10 - 4")
        assert result.success is True
        assert result.data["result"] == 6

    @pytest.mark.asyncio
    async def test_basic_multiplication(self):
        tool = CalculatorTool()
        result = await tool.safe_execute(expression="6 * 7")
        assert result.success is True
        assert result.data["result"] == 42

    @pytest.mark.asyncio
    async def test_basic_division(self):
        tool = CalculatorTool()
        result = await tool.safe_execute(expression="15 / 3")
        assert result.success is True
        assert result.data["result"] == 5.0

    @pytest.mark.asyncio
    async def test_complex_expression(self):
        tool = CalculatorTool()
        result = await tool.safe_execute(expression="(2 + 3) * 4")
        assert result.success is True
        assert result.data["result"] == 20

    @pytest.mark.asyncio
    async def test_power(self):
        tool = CalculatorTool()
        result = await tool.safe_execute(expression="2 ** 10")
        assert result.success is True
        assert result.data["result"] == 1024

    @pytest.mark.asyncio
    async def test_negative_numbers(self):
        tool = CalculatorTool()
        result = await tool.safe_execute(expression="-5 + 3")
        assert result.success is True
        assert result.data["result"] == -2

    @pytest.mark.asyncio
    async def test_division_by_zero(self):
        tool = CalculatorTool()
        result = await tool.safe_execute(expression="1 / 0")
        assert result.success is False
        assert "zero" in result.error.lower()

    @pytest.mark.asyncio
    async def test_empty_expression(self):
        tool = CalculatorTool()
        result = await tool.safe_execute(expression="")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_invalid_expression(self):
        tool = CalculatorTool()
        result = await tool.safe_execute(expression="hello world")
        assert result.success is False


class TestPythonTool:
    """Tests for PythonTool."""

    @pytest.mark.asyncio
    async def test_simple_print(self):
        tool = PythonTool()
        result = await tool.safe_execute(code="print('hello world')")
        assert result.success is True
        assert "hello world" in result.data["stdout"]

    @pytest.mark.asyncio
    async def test_calculation(self):
        tool = PythonTool()
        result = await tool.safe_execute(code="print(2 ** 10)")
        assert result.success is True
        assert "1024" in result.data["stdout"]

    @pytest.mark.asyncio
    async def test_error_handling(self):
        tool = PythonTool()
        result = await tool.safe_execute(code="1 / 0")
        assert result.success is False
        assert "ZeroDivisionError" in result.error

    @pytest.mark.asyncio
    async def test_empty_code(self):
        tool = PythonTool()
        result = await tool.safe_execute(code="")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_syntax_error(self):
        tool = PythonTool()
        result = await tool.safe_execute(code="def (")
        assert result.success is False


class TestShellTool:
    """Tests for ShellTool."""

    @pytest.mark.asyncio
    async def test_echo(self):
        tool = ShellTool()
        result = await tool.safe_execute(command="echo hello")
        assert result.success is True
        assert "hello" in result.data["stdout"]

    @pytest.mark.asyncio
    async def test_empty_command(self):
        tool = ShellTool()
        result = await tool.safe_execute(command="")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_blocked_command(self):
        tool = ShellTool()
        result = await tool.safe_execute(command="rm -rf /")
        assert result.success is False
        assert "blocked" in result.error.lower()

    @pytest.mark.asyncio
    async def test_exit_code(self):
        tool = ShellTool()
        result = await tool.safe_execute(command="exit 1")
        assert result.success is False
        assert result.data["return_code"] == 1


class TestFileTool:
    """Tests for FileTool."""

    @pytest.mark.asyncio
    async def test_write_and_read(self, tmp_path):
        tool = FileTool()
        test_file = tmp_path / "test.txt"

        # Write
        result = await tool.safe_execute(
            action="write", path=str(test_file), content="Hello, JARVIS!"
        )
        assert result.success is True

        # Read
        result = await tool.safe_execute(action="read", path=str(test_file))
        assert result.success is True
        assert result.data["content"] == "Hello, JARVIS!"

    @pytest.mark.asyncio
    async def test_read_nonexistent(self):
        tool = FileTool()
        result = await tool.safe_execute(action="read", path="/nonexistent/file.txt")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_list_directory(self, tmp_path):
        tool = FileTool()
        (tmp_path / "file1.txt").write_text("test")
        (tmp_path / "file2.txt").write_text("test")

        result = await tool.safe_execute(action="list", path=str(tmp_path))
        assert result.success is True
        assert result.data["count"] == 2

    @pytest.mark.asyncio
    async def test_file_info(self, tmp_path):
        tool = FileTool()
        test_file = tmp_path / "info_test.txt"
        test_file.write_text("test content")

        result = await tool.safe_execute(action="info", path=str(test_file))
        assert result.success is True
        assert result.data["is_file"] is True

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = FileTool()
        result = await tool.safe_execute(action="delete", path="/tmp/test")
        assert result.success is False


class TestSearchTool:
    """Tests for SearchTool."""

    @pytest.mark.asyncio
    async def test_empty_query(self):
        tool = SearchTool()
        result = await tool.safe_execute(query="")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_has_correct_metadata(self):
        tool = SearchTool()
        assert tool.name == "search"
        assert tool.category == "information"
        assert "query" in tool.required_params


class TestBrowserTool:
    """Tests for BrowserTool."""

    @pytest.mark.asyncio
    async def test_empty_url(self):
        tool = BrowserTool()
        result = await tool.safe_execute(url="")
        assert result.success is False

    def test_has_correct_metadata(self):
        tool = BrowserTool()
        assert tool.name == "browser"
        assert tool.category == "information"
        assert "url" in tool.required_params


class TestAPITool:
    """Tests for APITool."""

    @pytest.mark.asyncio
    async def test_empty_url(self):
        tool = APITool()
        result = await tool.safe_execute(method="GET", url="")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_invalid_method(self):
        tool = APITool()
        result = await tool.safe_execute(method="INVALID", url="https://example.com")
        assert result.success is False

    def test_has_correct_metadata(self):
        tool = APITool()
        assert tool.name == "api"
        assert tool.category == "integration"
