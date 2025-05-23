// Copyright © 2023-2024 Apple Inc.

#include <algorithm>
#include <numeric>
#include <sstream>
#include <unordered_set>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/array.h"
#include "mlx/compile.h"
#include "mlx/compile_impl.h"
#include "mlx/transforms.h"
#include "mlx/transforms_impl.h"
#include "mlx/utils.h"
#include "python/src/mlx_func.h"
#include "python/src/trees.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

// Needed for printing shapes and strides.
using mx::operator<<;

using IntOrVec = std::variant<int, std::vector<int>>;
using StrOrSet = std::variant<std::string, std::unordered_set<std::string>>;

inline std::string type_name_str(const nb::handle& o) {
  return nb::cast<std::string>(nb::type_name(o.type()));
}

auto validate_argnums_argnames(
    const std::optional<IntOrVec>& argnums,
    const StrOrSet& argnames) {
  std::unordered_set<std::string> setnames;
  if (auto pv = std::get_if<std::string>(&argnames); pv) {
    setnames = {*pv};
  } else {
    setnames = std::get<std::unordered_set<std::string>>(argnames);
  }

  if (!argnums.has_value()) {
    // argnums was not provided and argnames was empty
    if (setnames.empty()) {
      return std::make_pair(std::vector<int>{0}, setnames);
    } else {
      return std::make_pair(std::vector<int>{}, setnames);
    }
  }

  std::vector<int> vecnums;
  if (auto pv = std::get_if<int>(&(*argnums)); pv) {
    vecnums = {*pv};
  } else {
    vecnums = std::get<std::vector<int>>(*argnums);
  }

  return std::make_pair(vecnums, setnames);
}

auto py_value_and_grad(
    const nb::callable& fun,
    std::vector<int> argnums,
    std::unordered_set<std::string> argnames,
    const std::string& error_msg_tag,
    bool scalar_func_only) {
  // Sanitize argnums
  if (argnums.size() == 0 && argnames.size() == 0) {
    throw std::invalid_argument(
        error_msg_tag + " Gradient wrt no argument requested");
  }
  for (auto arg : argnums) {
    std::sort(argnums.begin(), argnums.end());
    if (argnums[0] < 0) {
      std::ostringstream msg;
      msg << error_msg_tag
          << " Can't compute the gradient of negative argument index "
          << argnums[0];
      throw std::invalid_argument(msg.str());
    }
    for (int i = 1; i < argnums.size(); ++i) {
      if (argnums[i] == argnums[i - 1]) {
        std::ostringstream msg;
        msg << error_msg_tag << " Duplicate argument index " << argnums[0]
            << " is not allowed.";
        throw std::invalid_argument(msg.str());
      }
    }
  }

  return [fun, argnums, argnames, error_msg_tag, scalar_func_only](
             nb::args& args, nb::kwargs& kwargs) {
    // Sanitize the input
    if (argnums.size() > 0 && argnums.back() >= args.size()) {
      std::ostringstream msg;
      msg << error_msg_tag << " Can't compute the gradient of argument index "
          << argnums.back() << " because the function is called with only "
          << args.size() << " positional arguments.";
      throw std::invalid_argument(msg.str());
    }

    for (auto& key : argnames) {
      if (!kwargs.contains(key)) {
        std::ostringstream msg;
        msg << error_msg_tag
            << " Can't compute the gradient of keyword argument '" << key
            << "' because the function is called with the "
            << "following keyword arguments {";
        for (auto item : kwargs) {
          msg << nb::cast<std::string>(item.first) << ",";
        }
        msg << "}";
        throw std::invalid_argument(msg.str());
      }
    }

    // Collect the arrays
    std::vector<mx::array> arrays;
    std::vector<int> counts(1, 0);
    std::vector<int> gradient_indices;
    for (int i = 0, j = 0; i < args.size(); ++i) {
      bool needs_grad = (j < argnums.size() && argnums[j] == i);
      auto argsi = tree_flatten(args[i], /* strict = */ needs_grad);
      if (needs_grad) {
        auto old_size = gradient_indices.size();
        gradient_indices.resize(old_size + argsi.size());
        std::iota(
            gradient_indices.begin() + old_size,
            gradient_indices.end(),
            arrays.size());
        j++;
        counts.push_back(argsi.size());
      }
      arrays.insert(arrays.end(), argsi.begin(), argsi.end());
    }
    for (auto item : kwargs) {
      bool needs_grad =
          (argnames.find(nb::cast<std::string>(item.first)) != argnames.end());
      auto argsk = tree_flatten(item.second, /* strict = */ needs_grad);
      if (needs_grad) {
        auto old_size = gradient_indices.size();
        gradient_indices.resize(old_size + argsk.size());
        std::iota(
            gradient_indices.begin() + old_size,
            gradient_indices.end(),
            arrays.size());
        counts.push_back(argsk.size());
      }
      arrays.insert(arrays.end(), argsk.begin(), argsk.end());
    }
    std::partial_sum(counts.cbegin(), counts.cend(), counts.begin());

    // value_out will hold the output of the python function in order to be
    // able to reconstruct the python tree of extra return values
    nb::object py_value_out;
    auto value_and_grads = mx::value_and_grad(
        [&fun,
         &arrays,
         &args,
         &kwargs,
         &py_value_out,
         &error_msg_tag,
         scalar_func_only](const std::vector<mx::array>& a) {
          nb::list tree;
          tree.append(args);
          tree.append(kwargs);
          tree_fill(tree, a);

          // Call the python function
          py_value_out = fun(*tree[0], **tree[1]);

          // Replace the tracers with the originals. Don't overwrite
          // locations which were written to during the call to fun
          int index = 0;
          tree_visit_update(tree, [&](nb::handle node) {
            auto replace_arr = nb::cast<mx::array>(node);
            if (replace_arr.id() == a[index].id()) {
              return nb::cast(arrays[index++]);
            } else {
              return nb::cast(replace_arr);
            }
          });

          // Validate the return value of the python function
          if (!nb::isinstance<mx::array>(py_value_out)) {
            if (scalar_func_only) {
              std::ostringstream msg;
              msg << error_msg_tag << " The return value of the function "
                  << "whose gradient we want to compute should be a "
                  << "scalar array; but " << type_name_str(py_value_out)
                  << " was returned.";
              throw std::invalid_argument(msg.str());
            }
            if (!nb::isinstance<nb::tuple>(py_value_out)) {
              std::ostringstream msg;
              msg << error_msg_tag << " The return value of the function "
                  << "whose gradient we want to compute should be either a "
                  << "scalar array or a tuple with the first value being a "
                  << "scalar array (Union[array, tuple[array, Any, ...]]); but "
                  << type_name_str(py_value_out) << " was returned.";
              throw std::invalid_argument(msg.str());
            }
            nb::tuple ret = nb::cast<nb::tuple>(py_value_out);
            if (ret.size() == 0) {
              std::ostringstream msg;
              msg << error_msg_tag << " The return value of the function "
                  << "whose gradient we want to compute should be either a "
                  << "scalar array or a non-empty tuple. The first value should be a "
                  << "scalar array and the rest can be anything. Instead, "
                  << "we got an empty tuple.";
              throw std::invalid_argument(msg.str());
            }
            if (!nb::isinstance<mx::array>(ret[0])) {
              std::ostringstream msg;
              msg << error_msg_tag << " The return value of the function "
                  << "whose gradient we want to compute should be either a "
                  << "scalar array or a tuple with the first value being a "
                  << "scalar array (Union[array, tuple[array, Any, ...]]); but it "
                  << "was a tuple with the first value being of type "
                  << type_name_str(ret[0]) << " .";
              throw std::invalid_argument(msg.str());
            }
          }

          return tree_flatten(py_value_out, false);
        },
        gradient_indices)(arrays);

    auto value = value_and_grads.first;
    auto gradients = value_and_grads.second;

    // Put the gradients back in their container.
    // We have the following cases:
    //
    // 1. Single python positional argument has a gradient (eg argnums=[0])
    // 2. Many python positional arguments have gradients (eg argnums=[0, 1])
    // 3. A python keyword argument has gradients
    //
    // In case 1 we return the original python variable but with the gradients.
    // In case 2 we return a tuple of the above.
    // In case 3 we return a tuple containing a tuple and dict (sth like
    // (tuple(), dict(x=mx.array(5))) ).
    nb::object positional_grads;
    nb::object keyword_grads;
    nb::object py_grads;

    // Collect the gradients for the positional arguments
    if (argnums.size() == 1) {
      positional_grads = tree_unflatten(args[argnums[0]], gradients, counts[0]);
    } else if (argnums.size() > 1) {
      nb::list grads_;
      for (int i = 0; i < argnums.size(); i++) {
        grads_.append(tree_unflatten(args[argnums[i]], gradients, counts[i]));
      }
      positional_grads = nb::tuple(grads_);
    } else {
      positional_grads = nb::none();
    }

    // No keyword argument gradients so return the tuple of gradients
    if (argnames.size() == 0) {
      py_grads = positional_grads;
    } else {
      nb::dict grads_;
      int i = 0;
      for (auto item : kwargs) {
        auto k = nb::cast<std::string>(item.first);
        if (argnames.find(k) != argnames.end()) {
          grads_[k.c_str()] = tree_unflatten(
              nb::borrow(item.second), gradients, counts[i++ + argnums.size()]);
        }
      }
      keyword_grads = grads_;

      py_grads = nb::make_tuple(positional_grads, keyword_grads);
    }

    // Put the values back in the container
    nb::object return_value = tree_unflatten(py_value_out, value);
    return std::make_pair(return_value, py_grads);
  };
}

auto py_vmap(
    const nb::callable& fun,
    const nb::object& in_axes,
    const nb::object& out_axes) {
  return [fun, in_axes, out_axes](const nb::args& args) {
    auto axes_to_flat_tree = [](const nb::object& tree,
                                const nb::object& axes,
                                bool output_axes) {
      std::vector<int> flat_axes;
      bool encountered_tuple = false;
      tree_visit(
          {tree, axes},
          [&flat_axes, &encountered_tuple, output_axes](
              const std::vector<nb::object>& inputs) {
            if (nb::isinstance<mx::array>(inputs[0])) {
              if (inputs[1].is_none()) {
                flat_axes.push_back(-1);
              } else if (nb::isinstance<nb::int_>(inputs[1])) {
                int axis = nb::cast<int>(nb::cast<nb::int_>(inputs[1]));
                const mx::array& x = nb::cast<mx::array>(inputs[0]);
                if (axis < 0) {
                  axis += x.ndim() + output_axes;
                }
                if (axis < 0 || axis >= (x.ndim() + output_axes)) {
                  std::ostringstream msg;
                  msg << "[vmap] Invalid" << (output_axes ? " output " : " ")
                      << "vectorization axis " << axis
                      << " for array with shape " << x.shape();
                  throw std::invalid_argument(msg.str());
                }
                flat_axes.push_back(axis);
              } else if (nb::isinstance<nb::tuple>(inputs[1])) {
                encountered_tuple = true;
                auto l = nb::cast<nb::tuple>(inputs[1]);
                if (l.size() == 1 && nb::isinstance<nb::int_>(l[0])) {
                  int axis = nb::cast<int>(nb::cast<nb::int_>(l[0]));
                  const mx::array& x = nb::cast<mx::array>(inputs[0]);
                  if (axis < 0) {
                    axis += x.ndim() + output_axes;
                  }
                  if (axis < 0 || axis >= (x.ndim() + output_axes)) {
                    std::ostringstream msg;
                    msg << "[vmap] Invalid" << (output_axes ? " output " : " ")
                        << "vectorization axis " << axis
                        << " for array with shape " << x.shape();
                    throw std::invalid_argument(msg.str());
                  }
                  flat_axes.push_back(axis);
                } else if (l.size() == 1 && l[0].is_none()) {
                  flat_axes.push_back(-1);
                } else {
                  throw std::invalid_argument(
                      "[vmap] axis must be int or None.");
                }
              } else {
                throw std::invalid_argument("[vmap] axis must be int or None.");
              }
            } else {
              throw std::invalid_argument(
                  "[vmap] The arguments should contain only arrays");
            }
          });
      if (encountered_tuple && !nb::isinstance<mx::array>(tree)) {
        throw std::invalid_argument("[vmap] axis must be int or None.");
      }
      return flat_axes;
    };

    // Inputs must be array or tree of arrays
    auto inputs = tree_flatten(args, true);
    auto flat_in_axes =
        axes_to_flat_tree((args.size() == 1) ? args[0] : args, in_axes, false);

    // py_value_out will hold the output of the python function in order to be
    // able to reconstruct the python tree of extra return values
    nb::object py_outputs;

    auto vmap_fn =
        [&fun, &args, &inputs, &py_outputs](const std::vector<mx::array>& a) {
          // Call the python function
          py_outputs = fun(*tree_unflatten(args, a));

          // Flatten the outputs
          return tree_flatten(py_outputs, true);
        };

    auto [trace_inputs, trace_outputs] =
        mx::detail::vmap_trace(vmap_fn, inputs, flat_in_axes);

    auto flat_out_axes = axes_to_flat_tree(py_outputs, out_axes, true);

    // Perform the vmap
    auto outputs = mx::detail::vmap_replace(
        inputs, trace_inputs, trace_outputs, flat_in_axes, flat_out_axes);

    // Put the outputs back in the container
    return tree_unflatten(py_outputs, outputs);
  };
}

std::unordered_map<std::uintptr_t, nb::object>& tree_cache() {
  // This map is used to Cache the tree structure of the outputs
  static std::unordered_map<std::uintptr_t, nb::object> tree_cache_;
  return tree_cache_;
}

struct PyCompiledFun {
  nb::callable fun;
  std::uintptr_t fun_id;
  nb::object captured_inputs;
  nb::object captured_outputs;
  bool shapeless;
  mutable size_t num_outputs{0};

  PyCompiledFun(
      const nb::callable& fun,
      nb::object inputs,
      nb::object outputs,
      bool shapeless)
      : fun(fun),
        fun_id(reinterpret_cast<std::uintptr_t>(fun.ptr())),
        captured_inputs(inputs),
        captured_outputs(outputs),
        shapeless(shapeless) {}

  PyCompiledFun(const PyCompiledFun&) = delete;
  PyCompiledFun& operator=(const PyCompiledFun&) = delete;
  PyCompiledFun& operator=(PyCompiledFun&& other) = delete;
  PyCompiledFun(PyCompiledFun&& other)
      : fun(std::move(other.fun)),
        fun_id(reinterpret_cast<std::uintptr_t>(fun.ptr())) {
    other.fun_id = 0;
    captured_inputs = std::move(other.captured_inputs);
    captured_outputs = std::move(other.captured_outputs);
    shapeless = other.shapeless;
    num_outputs = other.num_outputs;
  };

  nb::object call_impl(const nb::args& args, const nb::kwargs& kwargs) {
    // Flat array inputs
    std::vector<mx::array> inputs;

    // Compilation constants which includes the tree structure of the arguments
    std::vector<uint64_t> constants;

    // Reserve some large primes to signify the presence of an array, a list or
    // a dict in order to encode the structure of the pytree. We choose primes
    // to reduce slightly the chances of these numbers occurring by a
    // multiplication as values in the constants list.
    constexpr uint64_t array_identifier = 18446744073709551557UL;
    constexpr uint64_t list_identifier = 18446744073709551533UL;
    constexpr uint64_t dict_identifier = 18446744073709551521UL;

    // Flatten the tree with hashed constants and structure
    std::function<void(nb::handle)> recurse;
    recurse = [&](nb::handle obj) {
      if (nb::isinstance<nb::list>(obj)) {
        auto l = nb::cast<nb::list>(obj);
        constants.push_back(list_identifier);
        for (int i = 0; i < l.size(); ++i) {
          recurse(l[i]);
        }
      } else if (nb::isinstance<nb::tuple>(obj)) {
        auto l = nb::cast<nb::tuple>(obj);
        constants.push_back(list_identifier);
        for (auto item : obj) {
          recurse(item);
        }
      } else if (nb::isinstance<nb::dict>(obj)) {
        auto d = nb::cast<nb::dict>(obj);
        constants.push_back(dict_identifier);
        for (auto item : d) {
          auto r = item.first.attr("__hash__")();
          constants.push_back(nb::cast<int64_t>(r));
          recurse(item.second);
        }
      } else if (nb::isinstance<mx::array>(obj)) {
        inputs.push_back(nb::cast<mx::array>(obj));
        constants.push_back(array_identifier);
      } else if (nb::isinstance<nb::str>(obj)) {
        auto r = obj.attr("__hash__")();
        constants.push_back(nb::cast<int64_t>(r));
      } else if (nb::isinstance<nb::int_>(obj)) {
        constants.push_back(nb::cast<int64_t>(obj));
      } else if (nb::isinstance<nb::float_>(obj)) {
        auto r = nb::cast<double>(obj);
        constants.push_back(*reinterpret_cast<uint64_t*>(&r));
      } else {
        std::ostringstream msg;
        msg << "[compile] Function arguments must be trees of arrays "
            << "or constants (floats, ints, or strings), but received "
            << "type " << type_name_str(obj) << ".";
        throw std::invalid_argument(msg.str());
      }
    };

    recurse(args);
    int num_args = inputs.size();
    recurse(kwargs);
    auto compile_fun = [this, &args, &kwargs, num_args](
                           const std::vector<mx::array>& a) {
      // Put tracers into captured inputs
      std::vector<mx::array> flat_in_captures;
      std::vector<mx::array> trace_captures;
      if (!captured_inputs.is_none()) {
        flat_in_captures = tree_flatten(captured_inputs, false);
        trace_captures.insert(
            trace_captures.end(), a.end() - flat_in_captures.size(), a.end());
        tree_fill(captured_inputs, trace_captures);
      }

      auto tree_outputs =
          fun(*tree_unflatten(args, a), **tree_unflatten(kwargs, a, num_args));
      auto [outputs, py_outputs] =
          tree_flatten_with_structure(std::move(tree_outputs), false);

      tree_cache().insert({fun_id, py_outputs});

      num_outputs = outputs.size();
      if (!captured_outputs.is_none()) {
        auto flat_out_captures = tree_flatten(captured_outputs, false);
        outputs.insert(
            outputs.end(),
            std::make_move_iterator(flat_out_captures.begin()),
            std::make_move_iterator(flat_out_captures.end()));
      }

      // Replace tracers with originals in captured inputs
      if (!captured_inputs.is_none()) {
        tree_replace(captured_inputs, trace_captures, flat_in_captures);
      }
      return outputs;
    };

    if (!captured_inputs.is_none()) {
      auto flat_in_captures = tree_flatten(captured_inputs, false);
      inputs.insert(
          inputs.end(),
          std::make_move_iterator(flat_in_captures.begin()),
          std::make_move_iterator(flat_in_captures.end()));
    }

    // Compile and call
    auto outputs =
        mx::detail::compile(compile_fun, fun_id, shapeless, constants)(inputs);
    if (!captured_outputs.is_none()) {
      std::vector<mx::array> captures(
          std::make_move_iterator(outputs.begin() + num_outputs),
          std::make_move_iterator(outputs.end()));
      tree_fill(captured_outputs, captures);
    }

    // Put the outputs back in the container
    nb::object py_outputs = tree_cache().at(fun_id);
    return tree_unflatten_from_structure(py_outputs, outputs);
  }

  nb::object operator()(const nb::args& args, const nb::kwargs& kwargs) const {
    return const_cast<PyCompiledFun*>(this)->call_impl(args, kwargs);
  };

  ~PyCompiledFun() {
    nb::gil_scoped_acquire gil;

    tree_cache().erase(fun_id);
    mx::detail::compile_erase(fun_id);
    fun.reset();
    captured_inputs.reset();
    captured_outputs.reset();
  }
};

class PyCheckpointedFun {
 public:
  PyCheckpointedFun(nb::callable fun) : fun_(std::move(fun)) {}
  ~PyCheckpointedFun() {
    nb::gil_scoped_acquire gil;

    fun_.reset();
  }

  struct InnerFunction {
    nb::object fun_;
    nb::object args_structure_;
    std::weak_ptr<nb::object> output_structure_;

    InnerFunction(
        nb::object fun,
        nb::object args_structure,
        std::weak_ptr<nb::object> output_structure)
        : fun_(std::move(fun)),
          args_structure_(std::move(args_structure)),
          output_structure_(output_structure) {}
    ~InnerFunction() {
      nb::gil_scoped_acquire gil;

      fun_.reset();
      args_structure_.reset();
    }

    std::vector<mx::array> operator()(const std::vector<mx::array>& inputs) {
      auto args = nb::cast<nb::tuple>(
          tree_unflatten_from_structure(args_structure_, inputs));
      auto [outputs, output_structure] =
          tree_flatten_with_structure(fun_(*args[0], **args[1]), false);
      if (auto s = output_structure_.lock()) {
        *s = output_structure;
      }
      return outputs;
    }
  };

  nb::object call_impl(const nb::args& args, const nb::kwargs& kwargs) {
    auto output_structure = std::make_shared<nb::object>();
    auto full_args = nb::make_tuple(args, kwargs);
    auto [inputs, args_structure] =
        tree_flatten_with_structure(full_args, false);

    auto outputs = mx::checkpoint(
        InnerFunction(fun_, args_structure, output_structure))(inputs);

    return tree_unflatten_from_structure(*output_structure, outputs);
  }

  nb::object operator()(const nb::args& args, const nb::kwargs& kwargs) const {
    return const_cast<PyCheckpointedFun*>(this)->call_impl(args, kwargs);
  }

 private:
  nb::callable fun_;
};

int py_custom_function_tp_traverse(PyObject* self, visitproc visit, void* arg);

int py_custom_function_tp_clear(PyObject* self);

/**
 * PyCustomFunction is the class that implements the python decorator
 * `mx.custom_function`.
 *
 * It implements a callable that instead of simply calling `fun` it creates a
 * CustomTransforms primitive via the `custom_function` C++ op which allows us
 * to redefine the vjp, jvp and vmap transformations.
 *
 * The implementation is verbose due to explicit handling of the destruction of
 * various python objects to make sure that there is no double-free and that
 * all of them are deleted while under GIL.
 *
 * Namely, for every one of the functions passed to the C++ `custom_function`
 * we create a callable struct that holds the following python objects (when
 * needed).
 *
 *    - An nb::callable which holds the passed function or transform
 *    - An nb::object holding input structure, namely the `(args, kwargs)`
 *      passed to the function in order to be able to recreate the arguments
 *      from the input arrays.
 *    - A std::shared_ptr<nb::object> holding the output structure name the
 *      structure of the return value of `fun`. It is a shared_ptr so that it
 *      can be set when the function is called and then used in the `vjp`
 *      transform. We delete the object only when the shared_ptr is about to be
 *      deleted see `output_structure_.use_count() == 1` to make sure that the
 *      object is deleted under GIL.
 */
class PyCustomFunction {
 public:
  PyCustomFunction(nb::callable fun) : fun_(std::move(fun)) {}
  ~PyCustomFunction() {
    nb::gil_scoped_acquire gil;
    reset();
  }

  struct InnerFunction {
    nb::callable fun_;
    nb::object input_structure_;
    std::shared_ptr<nb::object> output_structure_;

    InnerFunction(
        nb::callable fun,
        nb::object input_structure,
        std::shared_ptr<nb::object> output_structure)
        : fun_(std::move(fun)),
          input_structure_(std::move(input_structure)),
          output_structure_(std::move(output_structure)) {}
    ~InnerFunction() {
      nb::gil_scoped_acquire gil;

      fun_.reset();
      input_structure_.reset();
      if (output_structure_.use_count() == 1) {
        output_structure_->reset();
      }
    }

    std::vector<mx::array> operator()(const std::vector<mx::array>& inputs) {
      nb::gil_scoped_acquire gil;

      auto new_inputs = nb::cast<nb::tuple>(
          tree_unflatten_from_structure(input_structure_, inputs));
      std::vector<mx::array> outputs;
      std::tie(outputs, *output_structure_) =
          tree_flatten_with_structure(fun_(*new_inputs[0], **new_inputs[1]));
      return outputs;
    }
  };

  struct InnerVJPFunction {
    nb::callable vjp_fun_;
    nb::object input_structure_;
    std::shared_ptr<nb::object> output_structure_;

    InnerVJPFunction(
        nb::callable vjp_fun,
        nb::object input_structure,
        std::shared_ptr<nb::object> output_structure)
        : vjp_fun_(std::move(vjp_fun)),
          input_structure_(std::move(input_structure)),
          output_structure_(std::move(output_structure)) {}
    ~InnerVJPFunction() {
      nb::gil_scoped_acquire gil;

      vjp_fun_.reset();
      input_structure_.reset();
      if (output_structure_.use_count() == 1) {
        output_structure_->reset();
      }
    }

    std::vector<mx::array> operator()(
        const std::vector<mx::array>& primals,
        const std::vector<mx::array>& cotangents,
        const std::vector<mx::array>& outputs) {
      nb::gil_scoped_acquire gil;

      auto new_inputs = nb::cast<nb::tuple>(
          tree_unflatten_from_structure(input_structure_, primals));
      auto args = nb::cast<nb::tuple>(new_inputs[0]);
      auto new_cotangents =
          tree_unflatten_from_structure(*output_structure_, cotangents);
      auto new_outputs =
          tree_unflatten_from_structure(*output_structure_, outputs);

      if (args.size() == 1) {
        return tree_flatten(
            vjp_fun_(args[0], new_cotangents, new_outputs, **new_inputs[1]),
            false);
      } else {
        return tree_flatten(
            vjp_fun_(args, new_cotangents, new_outputs, **new_inputs[1]),
            false);
      }
    }
  };

  struct InnerJVPFunction {
    nb::callable jvp_fun_;
    nb::object input_structure_;

    InnerJVPFunction(nb::callable jvp_fun, nb::object input_structure)
        : jvp_fun_(std::move(jvp_fun)),
          input_structure_(std::move(input_structure)) {}
    ~InnerJVPFunction() {
      nb::gil_scoped_acquire gil;

      jvp_fun_.reset();
      input_structure_.reset();
    }

    std::vector<mx::array> operator()(
        const std::vector<mx::array>& primals,
        const std::vector<mx::array>& tangents,
        const std::vector<int>& argnums) {
      nb::gil_scoped_acquire gil;

      auto new_inputs = nb::cast<nb::tuple>(
          tree_unflatten_from_structure(input_structure_, primals));
      auto args = nb::cast<nb::tuple>(new_inputs[0]);
      auto kwargs = nb::cast<nb::dict>(new_inputs[1]);
      if (kwargs.size() > 0) {
        throw std::invalid_argument(
            "[custom jvp] Function should only accept positional arguments");
      }

      // Make a new pytree which has tangents or None when a tangent is not
      // available.
      std::vector<bool> have_tangents(primals.size(), false);
      for (auto arg : argnums) {
        have_tangents[arg] = true;
      }
      int array_index = 0;
      int tangent_index = 0;
      auto new_tangents =
          nb::cast<nb::tuple>(tree_map(args, [&](nb::handle element) {
            if (nb::isinstance<mx::array>(element) &&
                have_tangents[array_index++]) {
              return nb::cast(tangents[tangent_index++]);
            } else {
              return nb::none();
            }
          }));

      if (args.size() == 1) {
        return tree_flatten(jvp_fun_(args[0], new_tangents[0]), false);
      } else {
        return tree_flatten(jvp_fun_(args, new_tangents), false);
      }
    }
  };

  struct InnerVmapFunction {
    nb::callable vmap_fun_;
    nb::object input_structure_;

    InnerVmapFunction(nb::callable vmap_fun, nb::object input_structure)
        : vmap_fun_(std::move(vmap_fun)),
          input_structure_(std::move(input_structure)) {}
    ~InnerVmapFunction() {
      nb::gil_scoped_acquire gil;

      vmap_fun_.reset();
      input_structure_.reset();
    }

    std::pair<std::vector<mx::array>, std::vector<int>> operator()(
        const std::vector<mx::array>& inputs,
        const std::vector<int>& axes) {
      nb::gil_scoped_acquire gil;

      auto new_inputs = nb::cast<nb::tuple>(
          tree_unflatten_from_structure(input_structure_, inputs));
      auto args = nb::cast<nb::tuple>(new_inputs[0]);
      auto kwargs = nb::cast<nb::dict>(new_inputs[1]);
      if (kwargs.size() > 0) {
        throw std::invalid_argument(
            "[custom vmap] Function should only accept positional arguments");
      }

      int arr_index = 0;
      auto new_axes =
          nb::cast<nb::tuple>(tree_map(args, [&](nb::handle element) {
            int axis = axes[arr_index++];
            if (nb::isinstance<mx::array>(element) && axis >= 0) {
              return nb::cast(axis);
            } else {
              return nb::none();
            }
          }));

      nb::object result;
      if (args.size() == 1) {
        result = vmap_fun_(args[0], new_axes[0]);
      } else {
        result = vmap_fun_(args, new_axes);
      }

      if (!nb::isinstance<nb::tuple>(result)) {
        throw std::invalid_argument(
            "[custom vmap] Vmap function should return a tuple with 2 items.");
      }
      nb::tuple result_tuple = nb::cast<nb::tuple>(result);
      if (result_tuple.size() != 2) {
        throw std::invalid_argument(
            "[custom vmap] Vmap function should return a tuple with 2 items.");
      }

      std::vector<mx::array> outputs;
      std::vector<int> output_axes;
      tree_visit({result_tuple[0], result_tuple[1]}, [&](auto objects) {
        if (nb::isinstance<mx::array>(objects[0])) {
          outputs.push_back(nb::cast<mx::array>(objects[0]));
          output_axes.push_back(
              objects[1].is_none() ? -1 : nb::cast<int>(objects[1]));
        }
      });

      return {outputs, output_axes};
    }
  };

  nb::object call_impl(const nb::args& args, const nb::kwargs& kwargs) {
    if (!vjp_fun_.has_value() && !jvp_fun_.has_value() &&
        !vmap_fun_.has_value()) {
      return fun_(*args, **kwargs);
    }

    // Extract the inputs and their structure in capturable vars
    std::vector<mx::array> input_arrays;
    nb::object input_structure;
    auto full_args = nb::make_tuple(args, kwargs);
    std::tie(input_arrays, input_structure) =
        tree_flatten_with_structure(full_args, false);

    // The output structure will be stored here to be used in the custom vjp
    // function
    auto output_structure = std::make_shared<nb::object>();

    // Make a function that calls fun_ in the forward pass and vjp_ in the
    // backward pass. Then call it immediately and return the results.
    auto f = mx::custom_function(
        InnerFunction(fun_, input_structure, output_structure),
        make_vjp_function(input_structure, output_structure),
        make_jvp_function(input_structure),
        make_vmap_function(input_structure));

    auto outputs = f(input_arrays);
    return tree_unflatten_from_structure(*output_structure, outputs);
  }

  PyCustomFunction& set_vjp(nb::callable vjp_fun) {
    vjp_fun_ = vjp_fun;
    return *this;
  }

  PyCustomFunction& set_jvp(nb::callable jvp_fun) {
    jvp_fun_ = jvp_fun;
    return *this;
  }

  PyCustomFunction& set_vmap(nb::callable vmap_fun) {
    vmap_fun_ = vmap_fun;
    return *this;
  }
  void reset() {
    fun_.reset();
    if (vjp_fun_.has_value()) {
      (*vjp_fun_).reset();
    }
    if (jvp_fun_.has_value()) {
      (*jvp_fun_).reset();
    }
    if (vmap_fun_.has_value()) {
      (*vmap_fun_).reset();
    }
  }

  friend int py_custom_function_tp_traverse(PyObject*, visitproc, void*);

 private:
  std::optional<InnerVJPFunction> make_vjp_function(
      nb::object input_structure,
      std::shared_ptr<nb::object> output_structure) {
    if (!vjp_fun_.has_value()) {
      return std::nullopt;
    }

    return InnerVJPFunction(*vjp_fun_, input_structure, output_structure);
  }

  std::optional<InnerJVPFunction> make_jvp_function(
      nb::object input_structure) {
    if (!jvp_fun_.has_value()) {
      return std::nullopt;
    }

    return InnerJVPFunction(*jvp_fun_, input_structure);
  }

  std::optional<InnerVmapFunction> make_vmap_function(
      nb::object input_structure) {
    if (!vmap_fun_.has_value()) {
      return std::nullopt;
    }

    return InnerVmapFunction(*vmap_fun_, input_structure);
  }

  nb::callable fun_;
  std::optional<nb::callable> vjp_fun_;
  std::optional<nb::callable> jvp_fun_;
  std::optional<nb::callable> vmap_fun_;
};

int py_custom_function_tp_traverse(PyObject* self, visitproc visit, void* arg) {
  Py_VISIT(Py_TYPE(self));
  if (!nb::inst_ready(self)) {
    return 0;
  }

  auto* p = nb::inst_ptr<PyCustomFunction>(self);
  nb::handle v = nb::find(p->fun_);
  Py_VISIT(v.ptr());
  if (p->vjp_fun_.has_value()) {
    nb::handle v = nb::find(*(p->vjp_fun_));
    Py_VISIT(v.ptr());
  }
  if (p->jvp_fun_.has_value()) {
    nb::handle v = nb::find(*(p->jvp_fun_));
    Py_VISIT(v.ptr());
  }
  if (p->vmap_fun_.has_value()) {
    nb::handle v = nb::find(*(p->vmap_fun_));
    Py_VISIT(v.ptr());
  }
  return 0;
}
int py_custom_function_tp_clear(PyObject* self) {
  auto* p = nb::inst_ptr<PyCustomFunction>(self);
  p->reset();
  return 0;
}
PyType_Slot py_custom_function_slots[] = {
    {Py_tp_traverse, (void*)py_custom_function_tp_traverse},
    {Py_tp_clear, (void*)py_custom_function_tp_clear},
    {0, 0}};

void init_transforms(nb::module_& m) {
  nb::class_<PyCustomFunction>(
      m,
      "custom_function",
      nb::type_slots(py_custom_function_slots),
      R"pbdoc(
      Set up a function for custom gradient and vmap definitions.

      This class is meant to be used as a function decorator. Instances are
      callables that behave identically to the wrapped function. However, when
      a function transformation is used (e.g. computing gradients using
      :func:`value_and_grad`) then the functions defined via
      :meth:`custom_function.vjp`, :meth:`custom_function.jvp` and
      :meth:`custom_function.vmap` are used instead of the default transformation.

      Note, all custom transformations are optional. Undefined transformations
      fall back to the default behaviour.

      Example:

        .. code-block:: python

            import mlx.core as mx

            @mx.custom_function
            def f(x, y):
                return mx.sin(x) * y

            @f.vjp
            def f_vjp(primals, cotangent, output):
                x, y = primals
                return cotan * mx.cos(x) * y, cotan * mx.sin(x)

            @f.jvp
            def f_jvp(primals, tangents):
              x, y = primals
              dx, dy = tangents
              return dx * mx.cos(x) * y + dy * mx.sin(x)

            @f.vmap
            def f_vmap(inputs, axes):
              x, y = inputs
              ax, ay = axes
              if ay != ax and ax is not None:
                  y = y.swapaxes(ay, ax)
              return mx.sin(x) * y, (ax or ay)

      All ``custom_function`` instances behave as pure functions. Namely, any
      variables captured will be treated as constants and no gradients will be
      computed with respect to the captured arrays. For instance:

        .. code-block:: python

          import mlx.core as mx

          def g(x, y):
            @mx.custom_function
            def f(x):
              return x * y

            @f.vjp
            def f_vjp(x, dx, fx):
              # Note that we have only x, dx and fx and nothing with respect to y
              raise ValueError("Abort!")

            return f(x)

          x = mx.array(2.0)
          y = mx.array(3.0)
          print(g(x, y))                     # prints 6.0
          print(mx.grad(g)(x, y))            # Raises exception
          print(mx.grad(g, argnums=1)(x, y)) # prints 0.0
      )pbdoc")
      .def(
          nb::init<nb::callable>(),
          "f"_a,
          nb::sig("def __init__(self, f: Callable)"))
      .def("__call__", &PyCustomFunction::call_impl)
      .def(
          "vjp",
          &PyCustomFunction::set_vjp,
          "f"_a,
          nb::sig("def vjp(self, f: Callable)"),
          R"pbdoc(
            Define a custom vjp for the wrapped function.

            The vjp function takes three arguments:

            - *primals*: A pytree that contains all the positional arguments to
              the function. It could be a single array, a tuple of arrays or a
              full blown tuple of dicts of arrays etc.
            - *cotangents*: A pytree that matches the structure of the output
              but contains the cotangents (usually the gradients of the loss
              function with respect to the outputs).
            - *outputs*: The outputs of the function to be used to avoid
              recomputing them for the gradient computation.

            The vjp function should return the same pytree structure as the
            primals but containing the corresponding computed cotangents.
          )pbdoc")
      .def(
          "jvp",
          &PyCustomFunction::set_jvp,
          "f"_a,
          nb::sig("def jvp(self, f: Callable)"),
          R"pbdoc(
            Define a custom jvp for the wrapped function.

            The jvp function takes two arguments:

            - *primals*: A pytree that contains all the positional arguments to
              the function. It could be a single array, a tuple of arrays or a
              full blown tuple of dicts of arrays etc.
            - *tangents*: A pytree that matches the structure of the inputs but
              instead contains the gradients wrt to each input. Tangents could
              be ``None`` if some inputs don't have an associated gradient.

            The jvp function should return the same pytree structure as the
            outputs of the function but containing the tangents.
          )pbdoc")
      .def(
          "vmap",
          &PyCustomFunction::set_vmap,
          "f"_a,
          nb::sig("def vmap(self, f: Callable)"),
          R"pbdoc(
            Define a custom vectorization transformation for the wrapped function.

            The vmap function takes two arguments:

            - *inputs*: A pytree that contains all the positional arguments to
              the function. It could be a single array, a tuple of arrays or a
              full blown tuple of dicts of arrays etc.
            - *axes*: A pytree that matches the structure of the inputs but
              instead contains the vectorization axis for each input or
              ``None`` if an input is not vectorized.

            The vmap function should return the outputs of the original
            function but vectorized over the provided axes. It should also
            return a pytree with the vectorization axes of each output. If some
            outputs are no longer vectorized, then their vectorization axis
            should be ``None``.
          )pbdoc");

  m.def(
      "eval",
      [](const nb::args& args) {
        std::vector<mx::array> arrays = tree_flatten(args, false);
        {
          nb::gil_scoped_release nogil;
          eval(arrays);
        }
      },
      nb::arg(),
      nb::sig("def eval(*args) -> None"),
      R"pbdoc(
        Evaluate an :class:`array` or tree of :class:`array`.

        Args:
            *args (arrays or trees of arrays): Each argument can be a single array
              or a tree of arrays. If a tree is given the nodes can be a Python
              :class:`list`, :class:`tuple` or :class:`dict`. Leaves which are not
              arrays are ignored.
      )pbdoc");
  m.def(
      "async_eval",
      [](const nb::args& args) {
        std::vector<mx::array> arrays = tree_flatten(args, false);
        {
          nb::gil_scoped_release nogil;
          async_eval(arrays);
        }
      },
      nb::arg(),
      nb::sig("def async_eval(*args)"),
      R"pbdoc(
        Asynchronously evaluate an :class:`array` or tree of :class:`array`.

        .. note::

          This is an experimental API and may change in future versions.

        Args:
            *args (arrays or trees of arrays): Each argument can be a single array
              or a tree of arrays. If a tree is given the nodes can be a Python
              :class:`list`, :class:`tuple` or :class:`dict`. Leaves which are not
              arrays are ignored.

        Example:
            >>> x = mx.array(1.0)
            >>> y = mx.exp(x)
            >>> mx.async_eval(y)
            >>> print(y)
            >>>
            >>> y = mx.exp(x)
            >>> mx.async_eval(y)
            >>> z = y + 3
            >>> mx.async_eval(z)
            >>> print(z)
      )pbdoc");
  m.def(
      "jvp",
      [](const nb::callable& fun,
         const std::vector<mx::array>& primals,
         const std::vector<mx::array>& tangents) {
        auto vfun = [&fun](const std::vector<mx::array>& primals) {
          auto out = fun(*nb::cast(primals));
          if (nb::isinstance<mx::array>(out)) {
            return std::vector<mx::array>{nb::cast<mx::array>(out)};
          } else {
            return nb::cast<std::vector<mx::array>>(out);
          }
        };
        return jvp(vfun, primals, tangents);
      },
      "fun"_a,
      "primals"_a,
      "tangents"_a,
      nb::sig(
          "def jvp(fun: Callable, primals: list[array], tangents: list[array]) -> tuple[list[array], list[array]]"),
      R"pbdoc(
        Compute the Jacobian-vector product.

        This computes the product of the Jacobian of a function ``fun`` evaluated
        at ``primals`` with the ``tangents``.

        Args:
            fun (Callable): A function which takes a variable number of :class:`array`
              and returns a single :class:`array` or list of :class:`array`.
            primals (list(array)): A list of :class:`array` at which to
              evaluate the Jacobian.
            tangents (list(array)): A list of :class:`array` which are the
              "vector" in the Jacobian-vector product. The ``tangents`` should be the
              same in number, shape, and type as the inputs of ``fun`` (i.e. the ``primals``).

        Returns:
            list(array): A list of the Jacobian-vector products which
            is the same in number, shape, and type of the inputs to ``fun``.
      )pbdoc");
  m.def(
      "vjp",
      [](const nb::callable& fun,
         const std::vector<mx::array>& primals,
         const std::vector<mx::array>& cotangents) {
        auto vfun = [&fun](const std::vector<mx::array>& primals) {
          auto out = fun(*nb::cast(primals));
          if (nb::isinstance<mx::array>(out)) {
            return std::vector<mx::array>{nb::cast<mx::array>(out)};
          } else {
            return nb::cast<std::vector<mx::array>>(out);
          }
        };
        return vjp(vfun, primals, cotangents);
      },
      "fun"_a,
      "primals"_a,
      "cotangents"_a,
      nb::sig(
          "def vjp(fun: Callable, primals: list[array], cotangents: list[array]) -> tuple[list[array], list[array]]"),
      R"pbdoc(
        Compute the vector-Jacobian product.

        Computes the product of the ``cotangents`` with the Jacobian of a
        function ``fun`` evaluated at ``primals``.

        Args:
          fun (Callable): A function which takes a variable number of :class:`array`
            and returns a single :class:`array` or list of :class:`array`.
          primals (list(array)): A list of :class:`array` at which to
            evaluate the Jacobian.
          cotangents (list(array)): A list of :class:`array` which are the
            "vector" in the vector-Jacobian product. The ``cotangents`` should be the
            same in number, shape, and type as the outputs of ``fun``.

        Returns:
            list(array): A list of the vector-Jacobian products which
            is the same in number, shape, and type of the outputs of ``fun``.
      )pbdoc");
  m.def(
      "value_and_grad",
      [](const nb::callable& fun,
         const std::optional<IntOrVec>& argnums,
         const StrOrSet& argnames) {
        auto [argnums_vec, argnames_set] =
            validate_argnums_argnames(argnums, argnames);
        return mlx_func(
            py_value_and_grad(
                fun, argnums_vec, argnames_set, "[value_and_grad]", false),
            fun);
      },
      "fun"_a,
      "argnums"_a = nb::none(),
      "argnames"_a = std::vector<std::string>{},
      nb::sig(
          "def value_and_grad(fun: Callable, argnums: Optional[Union[int, Sequence[int]]] = None, argnames: Union[str, Sequence[str]] = []) -> Callable"),
      R"pbdoc(
        Returns a function which computes the value and gradient of ``fun``.

        The function passed to :func:`value_and_grad` should return either
        a scalar loss or a tuple in which the first element is a scalar
        loss and the remaining elements can be anything.

        .. code-block:: python

            import mlx.core as mx

            def mse(params, inputs, targets):
                outputs = forward(params, inputs)
                lvalue = (outputs - targets).square().mean()
                return lvalue

            # Returns lvalue, dlvalue/dparams
            lvalue, grads = mx.value_and_grad(mse)(params, inputs, targets)

            def lasso(params, inputs, targets, a=1.0, b=1.0):
                outputs = forward(params, inputs)
                mse = (outputs - targets).square().mean()
                l1 = mx.abs(outputs - targets).mean()

                loss = a*mse + b*l1

                return loss, mse, l1

            (loss, mse, l1), grads = mx.value_and_grad(lasso)(params, inputs, targets)

        Args:
            fun (Callable): A function which takes a variable number of
              :class:`array` or trees of :class:`array` and returns
              a scalar output :class:`array` or a tuple the first element
              of which should be a scalar :class:`array`.
            argnums (int or list(int), optional): Specify the index (or indices)
              of the positional arguments of ``fun`` to compute the gradient
              with respect to. If neither ``argnums`` nor ``argnames`` are
              provided ``argnums`` defaults to ``0`` indicating ``fun``'s first
              argument.
            argnames (str or list(str), optional): Specify keyword arguments of
              ``fun`` to compute gradients with respect to. It defaults to [] so
              no gradients for keyword arguments by default.

        Returns:
            Callable: A function which returns a tuple where the first element
            is the output of `fun` and the second element is the gradients w.r.t.
            the loss.
      )pbdoc");
  m.def(
      "grad",
      [](const nb::callable& fun,
         const std::optional<IntOrVec>& argnums,
         const StrOrSet& argnames) {
        auto [argnums_vec, argnames_set] =
            validate_argnums_argnames(argnums, argnames);
        auto fn =
            py_value_and_grad(fun, argnums_vec, argnames_set, "[grad]", true);
        return mlx_func(
            [fn = std::move(fn)](nb::args& args, nb::kwargs& kwargs) {
              return fn(args, kwargs).second;
            },
            fun);
      },
      "fun"_a,
      "argnums"_a = nb::none(),
      "argnames"_a = std::vector<std::string>{},
      nb::sig(
          "def grad(fun: Callable, argnums: Optional[Union[int, Sequence[int]]] = None, argnames: Union[str, Sequence[str]] = []) -> Callable"),
      R"pbdoc(
        Returns a function which computes the gradient of ``fun``.

        Args:
            fun (Callable): A function which takes a variable number of
              :class:`array` or trees of :class:`array` and returns
              a scalar output :class:`array`.
            argnums (int or list(int), optional): Specify the index (or indices)
              of the positional arguments of ``fun`` to compute the gradient
              with respect to. If neither ``argnums`` nor ``argnames`` are
              provided ``argnums`` defaults to ``0`` indicating ``fun``'s first
              argument.
            argnames (str or list(str), optional): Specify keyword arguments of
              ``fun`` to compute gradients with respect to. It defaults to [] so
              no gradients for keyword arguments by default.

        Returns:
            Callable: A function which has the same input arguments as ``fun`` and
            returns the gradient(s).
      )pbdoc");
  m.def(
      "vmap",
      [](const nb::callable& fun,
         const nb::object& in_axes,
         const nb::object& out_axes) {
        return mlx_func(
            py_vmap(fun, in_axes, out_axes), fun, in_axes, out_axes);
      },
      "fun"_a,
      "in_axes"_a = 0,
      "out_axes"_a = 0,
      nb::sig(
          "def vmap(fun: Callable, in_axes: object = 0, out_axes: object = 0) -> Callable"),
      R"pbdoc(
        Returns a vectorized version of ``fun``.

        Args:
            fun (Callable): A function which takes a variable number of
              :class:`array` or a tree of :class:`array` and returns
              a variable number of :class:`array` or a tree of :class:`array`.
            in_axes (int, optional): An integer or a valid prefix tree of the
              inputs to ``fun`` where each node specifies the vmapped axis. If
              the value is ``None`` then the corresponding input(s) are not vmapped.
              Defaults to ``0``.
            out_axes (int, optional): An integer or a valid prefix tree of the
              outputs of ``fun`` where each node specifies the vmapped axis. If
              the value is ``None`` then the corresponding outputs(s) are not vmapped.
              Defaults to ``0``.

        Returns:
            Callable: The vectorized function.
      )pbdoc");
  m.def(
      "compile",
      [](const nb::callable& fun,
         const nb::object& inputs,
         const nb::object& outputs,
         bool shapeless) {
        //  Try to get the name
        auto n =
            nb::hasattr(fun, "__name__") ? fun.attr("__name__") : nb::none();
        auto name = n.is_none() ? "compiled"
                                : nb::cast<std::string>(fun.attr("__name__"));

        // Try to get the signature
        std::ostringstream sig;
        sig << "def " << name;
        auto inspect = nb::module_::import_("inspect");
        if (nb::cast<bool>(inspect.attr("isroutine")(fun))) {
          sig << nb::cast<std::string>(
              inspect.attr("signature")(fun).attr("__str__")());
        } else {
          sig << "(*args, **kwargs)";
        }

        // Try to get the doc string
        auto d = inspect.attr("getdoc")(fun);
        std::string doc =
            d.is_none() ? "MLX compiled function." : nb::cast<std::string>(d);

        auto sig_str = sig.str();
        return mlx_func(
            nb::cpp_function(
                PyCompiledFun{fun, inputs, outputs, shapeless},
                nb::name(name.c_str()),
                nb::sig(sig_str.c_str()),
                doc.c_str()),
            fun,
            inputs,
            outputs);
      },
      "fun"_a,
      "inputs"_a = nb::none(),
      "outputs"_a = nb::none(),
      "shapeless"_a = false,
      nb::sig(
          "def compile(fun: Callable, inputs: Optional[object] = None, outputs: Optional[object] = None, shapeless: bool = False) -> Callable"),
      R"pbdoc(
        Returns a compiled function which produces the same output as ``fun``.

        Args:
            fun (Callable): A function which takes a variable number of
              :class:`array` or trees of :class:`array` and returns
              a variable number of :class:`array` or trees of :class:`array`.
            inputs (list or dict, optional): These inputs will be captured during
              the function compilation along with the inputs to ``fun``. The ``inputs``
              can be a :obj:`list` or a :obj:`dict` containing arbitrarily nested
              lists, dictionaries, or arrays. Leaf nodes that are not
              :obj:`array` are ignored. Default: ``None``
            outputs (list or dict, optional): These outputs will be captured and
              updated in a compiled function. The ``outputs`` can be a
              :obj:`list` or a :obj:`dict` containing arbitrarily nested lists,
              dictionaries, or arrays. Leaf nodes that are not :obj:`array` are ignored.
              Default: ``None``
            shapeless (bool, optional): A function compiled with the ``shapeless``
              option enabled will not be recompiled when the input shape changes. Not all
              functions can be compiled with ``shapeless`` enabled. Attempting to compile
              such functions with shapeless enabled will throw. Note, changing the number
              of dimensions or type of any input will result in a recompilation even with
              ``shapeless`` set to ``True``. Default: ``False``

        Returns:
            Callable: A compiled function which has the same input arguments
            as ``fun`` and returns the the same output(s).
      )pbdoc");
  m.def(
      "disable_compile",
      &mx::disable_compile,
      R"pbdoc(
        Globally disable compilation. Setting the environment variable
        ``MLX_DISABLE_COMPILE`` can also be used to disable compilation.
      )pbdoc");
  m.def(
      "enable_compile",
      &mx::enable_compile,
      R"pbdoc(
        Globally enable compilation. This will override the environment
        variable ``MLX_DISABLE_COMPILE`` if set.
      )pbdoc");
  m.def(
      "checkpoint",
      [](nb::callable fun) { return mlx_func(PyCheckpointedFun{fun}, fun); },
      "fun"_a);

  // Register static Python object cleanup before the interpreter exits
  auto atexit = nb::module_::import_("atexit");
  atexit.attr("register")(nb::cpp_function([]() {
    tree_cache().clear();
    mx::detail::compile_clear_cache();
  }));
}
